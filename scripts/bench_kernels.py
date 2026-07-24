#!/usr/bin/env python
"""Per-module before/after micro-benchmark for ESMFold2 kernel swaps.

Isolates a single ESMFold2 op (trimul / transition / attention-pair-bias / ...),
builds it with real dimensions and synthetic inputs, and measures — for each
kernel *backend* — how **inference latency** and **peak GPU memory** change
relative to a pure-PyTorch baseline. Two latency numbers are reported:

  * eager      — a plain timed loop (includes CUDA kernel-launch overhead)
  * cuda-graph — capture + replay (launch overhead removed; the kernel's
                 true compute cost)

Every measurement is gated by an **accuracy check** (PASS/FAIL): each backend's
output is compared against a float32 reference, and the CUDA-graph replay output
is compared against the eager output. A speedup is only trustworthy when the
verdict is PASS — the model itself silently falls back to PyTorch when a kernel
fails to engage, and a mishandled graph capture can return stale/garbage fast.

The backends available *today* are the model's built-ins (``set_kernel_backend``):
``none`` (pure PyTorch), ``fused`` (vendored Triton, bf16-only), and
``cuequivariance`` (auto-skipped where it does not engage, e.g. sm86 / short L).
When a miniworld kernel is wired in later, add a backend entry to ``_BACKENDS``
whose ``apply`` swaps it in — the measurement code does not change.

Runs on a GPU compute node (use ``srun``), never the login node:

    srun -p gpu --gres=gpu:A6000:1 --mem=32G --cpus-per-task=8 \
      .venv/bin/python scripts/bench_kernels.py \
        --target trimul --backends none,fused --seq-len 256,512,768 \
        --dtype bf16 --graph --csv bench_trimul_a6000.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

# Allow running as `python scripts/bench_kernels.py` without installing the
# package: put the repo root (parent of scripts/) and scripts/ on the path so we
# can import the sibling run_esmfold2 helpers.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# TF32 + cueq-ops preload, reused from the run script (matches the paper config).
from run_esmfold2 import _enable_cueq_ops  # noqa: E402

try:
    from transformers.models.esmfold2.modeling_esmfold2 import PairTransition
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        CUE_AVAILABLE,
        TRITON_KERNELS_AVAILABLE,
        AdaptiveLayerNorm,
        AttentionPairBias,
        ConditionedTransitionBlock,
        OuterProductMean,
        MSAPairWeightedAveraging,
        PairUpdateBlock,
        Transition,
        TriangleMultiplicativeUpdate,
    )
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Could not import ESMFold2 modules from `transformers`. Run with the repo "
        "venv (.venv/bin/python) that ships transformers.models.esmfold2."
    ) from exc

_GB = 1024.0**3
_DTYPES = {"fp32": torch.float32, "float32": torch.float32,
           "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


# --------------------------------------------------------------------------- #
# Dimensions (defaults from configuration_esmfold2.py; CLI-overridable)
# --------------------------------------------------------------------------- #


@dataclass
class Dims:
    L: int  # sequence length (n_tokens)
    batch: int = 1
    d_pair: int = 256  # c_z
    d_token: int = 768  # c_token
    token_heads: int = 16
    transition_multiplier: int = 2
    d_msa: int = 128
    d_hidden: int = 32  # OPM hidden
    msa_depth: int = 128  # M (n MSA rows)
    msa_heads: int = 8
    msa_head_width: int = 32
    chunk_size: int | None = 64


# --------------------------------------------------------------------------- #
# Target registry
# --------------------------------------------------------------------------- #


@dataclass
class Target:
    """One benchmarkable op: how to build it, feed it, and run it."""

    name: str
    build: Callable[[Dims], nn.Module]
    make_inputs: Callable[[Dims, torch.dtype, torch.device], dict]
    run: Callable[[nn.Module, dict], torch.Tensor]
    # A/B-ready targets expose set_kernel_backend; baseline-only ones do not (yet).
    ab_ready: bool = True
    notes: str = ""

    def set_chunk(self, module: nn.Module, chunk: int | None) -> None:
        if hasattr(module, "set_chunk_size"):
            try:
                module.set_chunk_size(chunk)
            except Exception:
                pass


def _randn(shape, dtype, device, scale=1.0):
    return torch.randn(*shape, device=device, dtype=dtype) * scale


def _build_targets() -> dict[str, Target]:
    t: dict[str, Target] = {}

    # --- trimul (TriangleMultiplicativeUpdate, outgoing) --------------------- #
    def _mk_trimul_inputs(d: Dims, dt, dev):
        return {"z": _randn((d.batch, d.L, d.L, d.d_pair), dt, dev), "mask": None}

    t["trimul"] = Target(
        name="trimul",
        build=lambda d: TriangleMultiplicativeUpdate(dim=d.d_pair, _outgoing=True),
        make_inputs=_mk_trimul_inputs,
        run=lambda m, x: m(x["z"], mask=x["mask"]),
        notes="TriangleMultiplicativeUpdate(outgoing); mask=None. NOTE: fused-Triton "
        "only engages via pair_update_block; standalone trimul accelerates via cueq only.",
    )
    t["trimul_in"] = Target(
        name="trimul_in",
        build=lambda d: TriangleMultiplicativeUpdate(dim=d.d_pair, _outgoing=False),
        make_inputs=_mk_trimul_inputs,
        run=lambda m, x: m(x["z"], mask=x["mask"]),
        notes="TriangleMultiplicativeUpdate(incoming)",
    )

    # --- transition (LN + SwiGLU, pair-shaped) ------------------------------- #
    t["transition"] = Target(
        name="transition",
        build=lambda d: Transition(d.d_pair, expansion_ratio=4),
        make_inputs=lambda d, dt, dev: {
            "x": _randn((d.batch, d.L, d.L, d.d_pair), dt, dev)
        },
        run=lambda m, x: m(x["x"]),
        notes="Transition (pair) — fused LN+w12+SwiGLU on the fused backend",
    )

    # --- attention_pair_bias (diffusion token attention) -------------------- #
    t["attention_pair_bias"] = Target(
        name="attention_pair_bias",
        build=lambda d: AttentionPairBias(
            d_model=d.d_token, d_pair=d.d_pair, num_heads=d.token_heads,
            d_cond=d.d_token, use_conditioning=True,
        ),
        make_inputs=lambda d, dt, dev: {
            "a": _randn((d.batch, d.L, d.d_token), dt, dev),
            "s": _randn((d.batch, d.L, d.d_token), dt, dev),
            "z": _randn((d.batch, d.L, d.L, d.d_pair), dt, dev),
        },
        run=lambda m, x: m(x["a"], x["s"], x["z"], beta=0.0, attention_mask=None),
        notes="AttentionPairBias; beta=0, attention_mask=None (fused/cueq eligible)",
    )

    # --- pair_update_block (whole trunk block: trimul_out+in+transition) ---- #
    t["pair_update_block"] = Target(
        name="pair_update_block",
        build=lambda d: PairUpdateBlock(d_pair=d.d_pair, expansion_ratio=4),
        make_inputs=lambda d, dt, dev: {
            "pair": _randn((d.batch, d.L, d.L, d.d_pair), dt, dev),
        },
        run=lambda m, x: m(x["pair"], pair_attention_mask=None),
        notes="PairUpdateBlock = tri_mul_out + tri_mul_in + pair_transition",
    )

    # --- baseline-only (pure-PyTorch today; A/B once miniworld kernels land) - #
    t["adaln"] = Target(
        name="adaln",
        build=lambda d: AdaptiveLayerNorm(d.d_token, d.d_token, eps=1e-5),
        make_inputs=lambda d, dt, dev: {
            "a": _randn((d.batch, d.L, d.d_token), dt, dev),
            "s": _randn((d.batch, d.L, d.d_token), dt, dev),
        },
        run=lambda m, x: m(x["a"], x["s"]),
        ab_ready=False,
        notes="AdaptiveLayerNorm (adaLN-Zero); ~24 calls/diffusion-step",
    )
    t["conditioned_transition"] = Target(
        name="conditioned_transition",
        build=lambda d: ConditionedTransitionBlock(
            d.d_token, d.d_token, transition_multiplier=d.transition_multiplier,
            use_conditioning=True,
        ),
        make_inputs=lambda d, dt, dev: {
            "a": _randn((d.batch, d.L, d.d_token), dt, dev),
            "s": _randn((d.batch, d.L, d.d_token), dt, dev),
        },
        run=lambda m, x: m(x["a"], x["s"]),
        ab_ready=False,
        notes="ConditionedTransitionBlock (post-AdaLN SwiGLU + gate)",
    )
    t["msa_pair_weighted_avg"] = Target(
        name="msa_pair_weighted_avg",
        build=lambda d: MSAPairWeightedAveraging(
            d.d_msa, d.d_pair, n_heads=d.msa_heads, head_width=d.msa_head_width,
        ),
        make_inputs=lambda d, dt, dev: {
            "msa": _randn((d.batch, d.L, d.msa_depth, d.d_msa), dt, dev),
            "pair": _randn((d.batch, d.L, d.L, d.d_pair), dt, dev),
            "mask": torch.ones(d.batch, d.L, d.L, device=dev, dtype=torch.bool),
        },
        run=lambda m, x: m(x["msa"], x["pair"], x["mask"]),
        ab_ready=False,
        notes="MSAPairWeightedAveraging (bias-only gated attn, AF3 Alg.10)",
    )
    t["opm"] = Target(
        name="opm",
        build=lambda d: OuterProductMean(d.d_msa, d.d_hidden, d.d_pair),
        make_inputs=lambda d, dt, dev: {
            "m": _randn((d.batch, d.L, d.msa_depth, d.d_msa), dt, dev),
            "mask": torch.ones(d.batch, d.L, d.msa_depth, device=dev, dtype=dt),
        },
        run=lambda m, x: m(x["m"], x["mask"]),
        ab_ready=False,
        notes="OuterProductMean (chunked along i); O(L^2 d_hidden^2) transient",
    )
    return t


TARGETS = _build_targets()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


@dataclass
class Backend:
    name: str
    apply: Callable[[nn.Module], None]
    available: bool
    reason: str = ""


def _apply_builtin(module: nn.Module, backend: str | None) -> None:
    """Set a built-in kernel backend where the module supports it."""
    if hasattr(module, "set_kernel_backend"):
        module.set_kernel_backend(backend)


def resolve_backends(names: list[str], target: Target) -> list[Backend]:
    out: list[Backend] = []
    for name in names:
        if name == "none":
            out.append(Backend("none", lambda m: _apply_builtin(m, None), True))
        elif name == "fused":
            if not target.ab_ready:
                reason = "target is baseline-only (no set_kernel_backend yet)"
            elif not TRITON_KERNELS_AVAILABLE:
                reason = "vendored triton kernels not importable"
            else:
                reason = ""
            out.append(Backend("fused", lambda m: _apply_builtin(m, "fused"),
                               target.ab_ready and TRITON_KERNELS_AVAILABLE, reason))
        elif name == "cuequivariance":
            if not target.ab_ready:
                reason = "target is baseline-only (no set_kernel_backend yet)"
            elif not CUE_AVAILABLE:
                reason = "cuequivariance not importable"
            else:
                reason = ""
            out.append(Backend("cuequivariance",
                               lambda m: _apply_builtin(m, "cuequivariance"),
                               target.ab_ready and CUE_AVAILABLE, reason))
        else:
            raise ValueError(f"unknown backend {name!r}")
    # Baseline-only targets: only 'none' is meaningful.
    if not target.ab_ready:
        if not any(b.name == "none" for b in out):
            out.insert(0, Backend("none", lambda m: _apply_builtin(m, None), True))
    return out


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_eager(run, module, inputs, device, iters, warmup) -> list[float]:
    """Return per-iter latencies (ms). Uses CUDA events on GPU, perf_counter on CPU."""
    for _ in range(warmup):
        run(module, inputs)
    _sync(device)
    times: list[float] = []
    if device.type == "cuda":
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run(module, inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            run(module, inputs)
            times.append((time.perf_counter() - t0) * 1e3)
    return times


@dataclass
class GraphResult:
    status: str  # "ok" | "unsupported: <reason>" | "skipped"
    times_ms: list[float] = field(default_factory=list)
    replay_output: torch.Tensor | None = None
    static_inputs: dict | None = None
    graph: object | None = None


def capture_graph(run, module, inputs, device, warmup) -> GraphResult:
    """Capture a CUDA graph of one forward into static buffers. Returns a
    GraphResult; on any capture failure returns status='unsupported: ...'."""
    if device.type != "cuda":
        return GraphResult("skipped: non-cuda device")
    # Static input buffers (graphs replay from fixed addresses).
    static_inputs = {
        k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()
    }
    args_for = lambda: static_inputs
    try:
        # Warm up on a side stream before capture (required by the CUDA graph API;
        # also fixes Triton autotune configs so capture sees static launches).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(max(3, warmup)):
                run(module, args_for())
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = run(module, args_for())
        torch.cuda.synchronize()
        return GraphResult("ok", replay_output=static_out,
                           static_inputs=static_inputs, graph=g)
    except Exception as exc:  # capture not supported for this op/backend
        return GraphResult(f"unsupported: {type(exc).__name__}: {exc}")


def time_graph(gr: GraphResult, iters: int) -> list[float]:
    if gr.status != "ok" or gr.graph is None:
        return []
    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        gr.graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def peak_memory_gb(run, module, inputs, device) -> float:
    """Peak *allocated* bytes during a single eager forward (module already
    resident). Measured in eager mode — graphs pre-allocate a static pool."""
    if device.type != "cuda":
        return 0.0
    run(module, inputs)  # trigger any lazy allocation first
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run(module, inputs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / _GB


def _max_err(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    o = out.detach().float()
    r = ref.detach().float()
    diff = (o - r).abs()
    max_abs = float(diff.max())
    denom = r.abs().max().clamp(min=1e-12)
    max_rel = float((diff.max() / denom))
    return max_abs, max_rel


# --------------------------------------------------------------------------- #
# Per-config run
# --------------------------------------------------------------------------- #


@dataclass
class Row:
    target: str
    backend: str
    seq_len: int
    dtype: str
    chunk: str
    eager_ms_median: float
    graph_ms_median: float | None
    speedup_vs_baseline: float | None
    peak_gb: float
    delta_peak_gb: float | None
    max_abs_err: float
    max_rel_err: float
    accuracy_verdict: str
    graph_vs_eager_ok: str
    graph_status: str


def backend_dtype(backend_name: str, base: torch.dtype) -> torch.dtype:
    """Compute dtype each backend actually runs in. The vendored ``fused`` Triton
    kernels are bf16-only, and the model runs its pair track fp32-pytorch vs
    bf16-fused (the fused kernel does LayerNorm internally; the pure-PyTorch trimul
    upcasts the contraction to fp32, which mismatches bf16 LayerNorm weights). So
    ``fused`` is always bf16; ``none``/``cuequivariance`` use the requested base
    dtype (default fp32 — the model's real pure-PyTorch precision)."""
    if backend_name == "fused":
        return torch.bfloat16
    return base


def _tol_for(dtype: torch.dtype, atol_arg, rtol_arg) -> tuple[float, float]:
    if atol_arg is not None and rtol_arg is not None:
        return atol_arg, rtol_arg
    if dtype == torch.float32:
        return (atol_arg if atol_arg is not None else 1e-3,
                rtol_arg if rtol_arg is not None else 1e-3)
    return (atol_arg if atol_arg is not None else 2e-2,
            rtol_arg if rtol_arg is not None else 2e-2)


def bench_target(target: Target, args, device: torch.device) -> list[Row]:
    rows: list[Row] = []
    dtype = _DTYPES[args.dtype]
    backends = resolve_backends(args.backends.split(","), target)

    for L in [int(x) for x in str(args.seq_len).split(",")]:
        dims = Dims(L=L, batch=args.batch, msa_depth=args.msa_depth,
                    chunk_size=args.chunk_size)
        torch.manual_seed(args.seed)
        master = target.build(dims).to(device=device).eval().requires_grad_(False)
        target.set_chunk(master, args.chunk_size)

        # Inputs: one fp32 master set (fixed seed), cast per dtype so every
        # backend sees identical values.
        torch.manual_seed(args.seed + 1)
        inputs_fp32 = target.make_inputs(dims, torch.float32, device)

        def cast_inputs(dt):
            out = {}
            for k, v in inputs_fp32.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    out[k] = v.to(dt)
                else:
                    out[k] = v
            return out

        # float32 gold reference (backend=none, fp32 weights & inputs).
        ref_module = copy.deepcopy(master).to(torch.float32)
        target.set_chunk(ref_module, args.chunk_size)
        _apply_builtin(ref_module, None)
        with torch.no_grad():
            reference = target.run(ref_module, cast_inputs(torch.float32)).detach()
        del ref_module

        baseline_peak: float | None = None
        baseline_eager_med: float | None = None

        for be in backends:
            if not be.available:
                print(f"  [skip] backend={be.name}: {be.reason}")
                continue
            be_dtype = backend_dtype(be.name, dtype)
            try:
                rows.append(_bench_one(
                    target, be, be_dtype, dims, device, args, master,
                    cast_inputs, reference, baseline_eager_med, baseline_peak, L,
                ))
                if be.name == "none":
                    baseline_eager_med = rows[-1].eager_ms_median
                    baseline_peak = rows[-1].peak_gb
            except Exception as exc:  # one bad (backend,dtype) must not abort the sweep
                print(f"  [error] backend={be.name} L={L} dtype={be_dtype}: "
                      f"{type(exc).__name__}: {exc}")
                rows.append(Row(
                    target=target.name, backend=be.name, seq_len=L,
                    dtype=str(be_dtype).replace("torch.", ""), chunk=str(args.chunk_size),
                    eager_ms_median=float("nan"), graph_ms_median=None,
                    speedup_vs_baseline=None, peak_gb=float("nan"),
                    delta_peak_gb=None, max_abs_err=float("nan"),
                    max_rel_err=float("nan"), accuracy_verdict="ERROR",
                    graph_vs_eager_ok="-", graph_status=f"{type(exc).__name__}: {exc}",
                ))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def _bench_one(target, be, be_dtype, dims, device, args, master, cast_inputs,
               reference, baseline_eager_med, baseline_peak, L) -> "Row":
    """Measure one (target, backend) at its natural dtype; return one Row.

    ``baseline_eager_med`` / ``baseline_peak`` are the ``none``-backend numbers
    (None until it has run) used for the speedup / Δpeak columns.
    """
    module = copy.deepcopy(master).to(be_dtype)
    target.set_chunk(module, args.chunk_size)
    be.apply(module)
    module.eval().requires_grad_(False)
    inputs = cast_inputs(be_dtype)

    with torch.no_grad():
        # Accuracy vs the fp32 reference.
        out = target.run(module, inputs).detach()
        max_abs, max_rel = _max_err(out, reference)
        atol, rtol = _tol_for(be_dtype, args.atol, args.rtol)
        verdict = "PASS" if max_abs <= atol + rtol * float(
            reference.abs().max()) else "FAIL"

        # Eager latency + peak memory.
        eager = time_eager(target.run, module, inputs, device, args.iters, args.warmup)
        eager_med = statistics.median(eager)
        peak = peak_memory_gb(target.run, module, inputs, device)

        # CUDA-graph latency + graph-vs-eager correctness.
        graph_med: float | None = None
        gstatus = "skipped"
        gve_ok = "-"
        if args.graph and device.type == "cuda":
            gr = capture_graph(target.run, module, inputs, device, args.warmup)
            gstatus = gr.status
            if gr.status == "ok":
                gtimes = time_graph(gr, args.iters)
                graph_med = statistics.median(gtimes) if gtimes else None
                for k, v in inputs.items():
                    if torch.is_tensor(v) and gr.static_inputs.get(k) is not None:
                        gr.static_inputs[k].copy_(v)
                gr.graph.replay()
                torch.cuda.synchronize()
                gdiff, _ = _max_err(gr.replay_output, out)
                gtol = 1e-5 if be_dtype == torch.float32 else 2e-2
                gve_ok = "True" if gdiff <= gtol else f"False({gdiff:.2e})"

    speedup = (baseline_eager_med / eager_med) if baseline_eager_med else None
    delta_peak = (peak - baseline_peak) if baseline_peak is not None else None
    return Row(
        target=target.name, backend=be.name, seq_len=L,
        dtype=str(be_dtype).replace("torch.", ""), chunk=str(args.chunk_size),
        eager_ms_median=round(eager_med, 4),
        graph_ms_median=round(graph_med, 4) if graph_med is not None else None,
        speedup_vs_baseline=round(speedup, 3) if speedup else None,
        peak_gb=round(peak, 4),
        delta_peak_gb=round(delta_peak, 4) if delta_peak is not None else None,
        max_abs_err=round(max_abs, 6), max_rel_err=round(max_rel, 6),
        accuracy_verdict=verdict, graph_vs_eager_ok=gve_ok, graph_status=gstatus,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_table(rows: list[Row]) -> None:
    hdr = (f"{'target':<22}{'backend':<15}{'L':>5}{'dtype':>7}"
           f"{'eager(ms)':>11}{'graph(ms)':>11}{'speedup':>9}"
           f"{'peak(GB)':>10}{'Δpeak':>9}{'maxAbsErr':>11}{'acc':>6}{'g==e':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        g = f"{r.graph_ms_median:.3f}" if r.graph_ms_median is not None else "-"
        sp = f"{r.speedup_vs_baseline:.2f}x" if r.speedup_vs_baseline else "-"
        dp = f"{r.delta_peak_gb:+.3f}" if r.delta_peak_gb is not None else "-"
        flag = "  <== FAIL" if r.accuracy_verdict == "FAIL" else ""
        print(f"{r.target:<22}{r.backend:<15}{r.seq_len:>5}{r.dtype:>7}"
              f"{r.eager_ms_median:>11.3f}{g:>11}{sp:>9}"
              f"{r.peak_gb:>10.3f}{dp:>9}{r.max_abs_err:>11.2e}"
              f"{r.accuracy_verdict:>6}{r.graph_vs_eager_ok:>10}{flag}")


def write_csv(path: Path, rows: list[Row]) -> None:
    fields = list(Row.__dataclass_fields__.keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Per-module before/after kernel micro-benchmark for ESMFold2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", default="trimul",
                   help="Target op name, comma-separated list, or 'all'.")
    p.add_argument("--backends", default="none,fused",
                   help="Comma list of none,fused,cuequivariance.")
    p.add_argument("--seq-len", default="256,512", help="Comma list of L to sweep.")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="fp32", choices=list(_DTYPES),
                   help="Baseline/reference dtype for none & cueq (default fp32, the "
                   "model's real pure-PyTorch precision). The fused backend always "
                   "runs bf16; a bf16 baseline is invalid for trimul (fp32-upcast LN).")
    p.add_argument("--msa-depth", type=int, default=128,
                   help="MSA rows (M) for opm / msa_pair_weighted_avg targets.")
    p.add_argument("--chunk-size", type=lambda s: None if s.lower() == "none" else int(s),
                   default=64, help="Chunk size for trimul/transition/opm (or 'none').")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--graph", action="store_true", help="Also capture+time a CUDA graph.")
    p.add_argument("--no-graph", dest="graph", action="store_false")
    p.set_defaults(graph=True)
    p.add_argument("--atol", type=float, default=None, help="Accuracy atol override.")
    p.add_argument("--rtol", type=float, default=None, help="Accuracy rtol override.")
    p.add_argument("--fail-on-accuracy", action="store_true", default=True,
                   help="Exit non-zero if any backend FAILs the accuracy gate.")
    p.add_argument("--no-fail-on-accuracy", dest="fail_on_accuracy",
                   action="store_false")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--list-targets", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_targets:
        print("Available targets:")
        for name, tgt in TARGETS.items():
            tag = "A/B-ready" if tgt.ab_ready else "baseline-only"
            print(f"  {name:<24}[{tag}]  {tgt.notes}")
        return 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _enable_cueq_ops()

    device = torch.device(args.device)
    print(f"device={device} dtype={args.dtype} backends={args.backends} "
          f"seq_len={args.seq_len} chunk={args.chunk_size} "
          f"(triton={TRITON_KERNELS_AVAILABLE}, cueq={CUE_AVAILABLE})")

    if args.target == "all":
        names = list(TARGETS)
    else:
        names = [n.strip() for n in args.target.split(",")]
    for n in names:
        if n not in TARGETS:
            raise SystemExit(f"unknown target {n!r}; see --list-targets")

    all_rows: list[Row] = []
    for n in names:
        print(f"\n=== target: {n} ({TARGETS[n].notes}) ===")
        all_rows.extend(bench_target(TARGETS[n], args, device))

    print_table(all_rows)
    if args.csv is not None:
        write_csv(args.csv, all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.csv}")

    fails = [r for r in all_rows if r.accuracy_verdict == "FAIL"]
    gfails = [r for r in all_rows if r.graph_vs_eager_ok.startswith("False")]
    if fails or gfails:
        print(f"\nACCURACY: {len(fails)} backend(s) FAILed parity, "
              f"{len(gfails)} graph!=eager.")
        if args.fail_on_accuracy:
            return 1
    else:
        print("\nACCURACY: all backends PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
