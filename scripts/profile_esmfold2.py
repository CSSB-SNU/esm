#!/usr/bin/env python
"""Profile ESMFold2 inference: per-module wall time + peak GPU memory.

ESMFold2 runs these heavy stages inside ``ESMFold2Model.forward()``:

  1. ESM-C language model  — the ESMC-6B backbone (``model._esmc``) plus the
     pair projection shim (``model.language_model``).
  2. Pair trunk            — the recycling loop over ``model.folding_trunk``
     (``PairUpdateBlock``s), plus ``model.msa_encoder``.
  3. Diffusion trunk       — ``model.structure_head.sample(...)``, an iterative
     denoiser running ``num_sampling_steps`` steps of ``diffusion_module``.

This script instruments those stages **non-invasively** — it attaches PyTorch
forward hooks / monkey-patches bound methods at runtime, and never edits the
installed ``transformers`` package. It reuses the loading / input-building
machinery from ``run_esmfold2.py``.

Two modes:

  * default (step 1) — coarse: the 3 modules above.
  * ``--deep`` (step 2) — per ``folding_trunk`` loop iteration, per
    ``PairUpdateBlock``, and per diffusion sampling step.

Example
-------
    srun --gres=gpu:1 --pty python scripts/profile_esmfold2.py \
        --input job.json --dtype bfloat16
    srun --gres=gpu:1 --pty python scripts/profile_esmfold2.py \
        --input H2343.json --dtype bfloat16 --deep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

# Allow running as `python scripts/profile_esmfold2.py` without installing the
# package: put the repo root (parent of scripts/) on the import path, and this
# scripts/ dir so we can import the sibling run_esmfold2 helpers.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from run_esmfold2 import (  # noqa: E402
    _DTYPES,
    LMOffloader,
    build_input,
    install_lm_offload,
    load_job,
    load_model,
)

_MB = 1024.0**2
_GB = 1024.0**3


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #


class _Record:
    """Accumulated measurements for one labelled region."""

    __slots__ = ("times", "peaks")

    def __init__(self) -> None:
        self.times: list[float] = []  # seconds, one per call
        self.peaks: list[int] = []  # bytes, only for top-level (depth-0) calls

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def total_time(self) -> float:
        return sum(self.times)

    @property
    def peak(self) -> int:
        return max(self.peaks) if self.peaks else 0


class ModuleProfiler:
    """Times labelled regions and attributes a GPU peak to each top-level region.

    Memory is device-global, so a per-region peak is only meaningful for the
    *outermost* active region: nested regions (e.g. ``PairUpdateBlock`` inside
    ``folding_trunk``) are timed but not memory-attributed, because resetting the
    peak counter inside them would corrupt the enclosing region's measurement. A
    depth counter enforces this: ``reset_peak_memory_stats`` runs only when
    entering at depth 0, and ``max_memory_allocated`` is read only when returning
    to depth 0.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.records: dict[str, _Record] = {}
        self._depth = 0
        self._stack: list[tuple[str, float]] = []
        self._handles: list = []  # forward-hook handles to remove
        self._patches: list[tuple[object, str, object]] = []  # (obj, attr, orig)
        # Chrome Trace Event Format capture (loadable in chrome://tracing / Perfetto)
        self.tracing = False
        self._trace_origin = 0.0
        self.events: list[dict] = []  # complete ("X") events with ts/dur in µs

    def enable_trace(self) -> None:
        self.tracing = True
        self._trace_origin = time.perf_counter()

    # -- low-level region bracket ------------------------------------------- #

    def _enter(self, label: str) -> None:
        if self.cuda:
            torch.cuda.synchronize()
            if self._depth == 0:
                torch.cuda.reset_peak_memory_stats()
        self._depth += 1
        self._stack.append((label, time.perf_counter()))

    def _exit(self) -> None:
        label, t0 = self._stack.pop()
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        elapsed = now - t0
        self._depth -= 1
        rec = self.records.setdefault(label, _Record())
        rec.times.append(elapsed)
        if self._depth == 0 and self.cuda:
            rec.peaks.append(torch.cuda.max_memory_allocated())
        if self.tracing:
            # Same pid/tid for all → nested ts/dur render as a flame chart.
            self.events.append(
                {
                    "name": label,
                    "ph": "X",
                    "ts": (t0 - self._trace_origin) * 1e6,
                    "dur": elapsed * 1e6,
                    "pid": 0,
                    "tid": 0,
                }
            )

    @contextmanager
    def region(self, label: str):
        self._enter(label)
        try:
            yield
        finally:
            self._exit()

    # -- attaching to a model ----------------------------------------------- #

    def hook_module(self, module: torch.nn.Module, label: str) -> None:
        """Time a module's forward via pre/post hooks (depth-aware memory)."""
        if module is None:
            return

        def pre_hook(_mod, _args, _kwargs):
            self._enter(label)

        def post_hook(_mod, _args, _kwargs, _out):
            self._exit()

        self._handles.append(
            module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )
        self._handles.append(
            module.register_forward_hook(post_hook, with_kwargs=True)
        )

    def patch_method(self, obj: object, attr: str, label: str) -> None:
        """Wrap a plain (non-forward) bound method as a timed region."""
        orig = getattr(obj, attr)
        profiler = self

        def wrapped(*args, **kwargs):
            with profiler.region(label):
                return orig(*args, **kwargs)

        setattr(obj, attr, wrapped)
        self._patches.append((obj, attr, orig))

    def timer_method(self, obj: object, attr: str, label: str) -> None:
        """Wrap a method as a pure wall-clock timer (no depth / no memory).

        Used for the top-level ``model.forward`` so it does not become the sole
        owner of device peak memory and starve the inner regions.
        """
        orig = getattr(obj, attr)
        records = self.records
        cuda = self.cuda

        def wrapped(*args, **kwargs):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                if cuda:
                    torch.cuda.synchronize()
                records.setdefault(label, _Record()).times.append(
                    time.perf_counter() - t0
                )

        setattr(obj, attr, wrapped)
        self._patches.append((obj, attr, orig))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for obj, attr, orig in reversed(self._patches):
            setattr(obj, attr, orig)
        self._patches.clear()

    def reset_measurements(self) -> None:
        self.records.clear()
        self._depth = 0
        self._stack.clear()


# Region labels (kept stable so the report / CSV columns are predictable).
LM_ESMC = "esm-c backbone (_esmc)"
LM_SHIM = "esm-c shim (language_model)"
PAIR_TRUNK = "pair trunk (folding_trunk)"
MSA_ENCODER = "msa_encoder"
DIFFUSION = "diffusion trunk (structure_head.sample)"
DIFFUSION_STEP = "diffusion step (diffusion_module)"
MODEL_FORWARD = "model.forward (total)"


def install_step1(prof: ModuleProfiler, model: torch.nn.Module) -> None:
    """Coarse 3-module instrumentation."""
    prof.timer_method(model, "forward", MODEL_FORWARD)
    prof.hook_module(getattr(model, "_esmc", None), LM_ESMC)
    prof.hook_module(getattr(model, "language_model", None), LM_SHIM)
    prof.hook_module(getattr(model, "folding_trunk", None), PAIR_TRUNK)
    prof.hook_module(getattr(model, "msa_encoder", None), MSA_ENCODER)
    # Diffusion: sample() is a plain method, and we do NOT hook diffusion_module
    # in step 1, so the sample region is top-level and owns the diffusion peak.
    prof.patch_method(model.structure_head, "sample", DIFFUSION)


def install_deep(prof: ModuleProfiler, model: torch.nn.Module) -> None:
    """Step-2 instrumentation: per-iteration / per-block / per-step detail."""
    prof.timer_method(model, "forward", MODEL_FORWARD)
    prof.hook_module(getattr(model, "_esmc", None), LM_ESMC)
    prof.hook_module(getattr(model, "language_model", None), LM_SHIM)

    # Pair trunk: folding_trunk is top-level (per-iteration peak); its blocks
    # are nested -> timed only, labelled by block index.
    trunk = getattr(model, "folding_trunk", None)
    prof.hook_module(trunk, PAIR_TRUNK)
    if trunk is not None and hasattr(trunk, "blocks"):
        for i, block in enumerate(trunk.blocks):
            prof.hook_module(block, f"block[{i:02d}]")

    # MSA encoder: top-level (per-iteration peak) + per-MSAEncoderBlock (nested).
    msa = getattr(model, "msa_encoder", None)
    prof.hook_module(msa, MSA_ENCODER)
    if msa is not None and hasattr(msa, "blocks"):
        for i, block in enumerate(msa.blocks):
            prof.hook_module(block, f"msa_block[{i:02d}]")

    # Diffusion: hook diffusion_module directly (top-level per step), and do NOT
    # wrap sample(), so each step gets its own time + peak. Total diffusion time
    # is the sum over steps.
    prof.hook_module(getattr(model.structure_head, "diffusion_module", None),
                     DIFFUSION_STEP)


# --------------------------------------------------------------------------- #
# torch.profiler region labels (for the GPU-kernel + memory trace)
# --------------------------------------------------------------------------- #


class RecordFunctionLabeler:
    """Wraps modules in ``torch.profiler.record_function`` ranges via forward
    hooks, so they appear as named user annotations in the trace — correlated
    (flow arrows) to the GPU kernels they launch. Non-invasive: no edits to the
    installed model code. Stack-based so nested modules nest correctly.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._stack: list = []

    def wrap(self, module: torch.nn.Module | None, label: str) -> None:
        if module is None:
            return

        def pre_hook(_m, _args):
            rf = torch.profiler.record_function(label)
            rf.__enter__()
            self._stack.append(rf)

        def post_hook(_m, _args, _out):
            self._stack.pop().__exit__(None, None, None)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def remove(self) -> None:
        # Close any still-open ranges (e.g. on error), then detach hooks.
        while self._stack:
            self._stack.pop().__exit__(None, None, None)
        for h in self._handles:
            h.remove()
        self._handles.clear()


def install_record_labels(labeler: RecordFunctionLabeler, model: torch.nn.Module) -> None:
    """Label the same regions as install_deep, for the torch.profiler trace."""
    labeler.wrap(getattr(model, "_esmc", None), "esm-c")
    trunk = getattr(model, "folding_trunk", None)
    labeler.wrap(trunk, "pair_trunk")
    if trunk is not None and hasattr(trunk, "blocks"):
        for i, block in enumerate(trunk.blocks):
            labeler.wrap(block, f"PairUpdateBlock[{i:02d}]")
    msa = getattr(model, "msa_encoder", None)
    labeler.wrap(msa, "msa_encoder")
    if msa is not None and hasattr(msa, "blocks"):
        for i, block in enumerate(msa.blocks):
            labeler.wrap(block, f"MSAEncoderBlock[{i:02d}]")
    labeler.wrap(getattr(model.structure_head, "diffusion_module", None), "diffusion_step")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:8.1f}"


def _fmt_gb(nbytes: int) -> str:
    return f"{nbytes / _GB:7.2f}"


def print_step1_table(prof: ModuleProfiler, baseline_bytes: int) -> None:
    fwd = prof.records.get(MODEL_FORWARD)
    fwd_time = fwd.total_time if fwd else 0.0

    order = [LM_ESMC, LM_SHIM, PAIR_TRUNK, MSA_ENCODER, DIFFUSION]
    print("\n=== Step 1: per-module time & peak GPU memory ===")
    print(f"baseline (resident after load): {baseline_bytes / _GB:.2f} GB")
    print(
        f"{'module':<40}{'time(ms)':>10}{'%fwd':>7}{'calls':>7}"
        f"{'mean(ms)':>10}{'peak(GB)':>10}{'Δpeak(GB)':>11}"
    )
    print("-" * 95)
    module_sum = 0.0
    for label in order:
        rec = prof.records.get(label)
        if rec is None or rec.count == 0:
            continue
        module_sum += rec.total_time
        pct = 100.0 * rec.total_time / fwd_time if fwd_time else 0.0
        mean_ms = rec.total_time / rec.count * 1e3
        peak = rec.peak
        dpeak = max(0, peak - baseline_bytes)
        peak_s = _fmt_gb(peak) if peak else "      -"
        dpeak_s = _fmt_gb(dpeak) if peak else "      -"
        print(
            f"{label:<40}{_fmt_ms(rec.total_time)}{pct:6.1f}%{rec.count:7d}"
            f"{mean_ms:10.2f}{peak_s}{dpeak_s}"
        )
    print("-" * 95)
    print(f"{'sum of measured modules':<40}{_fmt_ms(module_sum)}")
    print(f"{'model.forward total':<40}{_fmt_ms(fwd_time)}")
    gap = fwd_time - module_sum
    print(
        f"{'unmeasured glue (embed/pos/disto/conf)':<40}{_fmt_ms(gap)}"
        f"{100.0 * gap / fwd_time if fwd_time else 0.0:6.1f}%"
    )


def _summarize(times: list[float]) -> str:
    ms = [t * 1e3 for t in times]
    if not ms:
        return "n/a"
    if len(ms) == 1:
        return f"{ms[0]:.2f}ms"
    return (
        f"sum={sum(ms):.1f}  mean={statistics.mean(ms):.2f}  "
        f"min={min(ms):.2f}  max={max(ms):.2f}ms (n={len(ms)})"
    )


def print_deep_report(prof: ModuleProfiler, baseline_bytes: int) -> None:
    print("\n=== Step 2 (deep): pair trunk ===")
    trunk = prof.records.get(PAIR_TRUNK)
    if trunk and trunk.count:
        print(f"folding_trunk per loop iteration: {_summarize(trunk.times)}")
        if trunk.peaks:
            print(
                f"  per-iter peak GPU: max {_fmt_gb(max(trunk.peaks)).strip()} GB  "
                f"(Δ over baseline {max(0, max(trunk.peaks) - baseline_bytes) / _GB:.2f} GB)"
            )
        # Per-block breakdown (mean ms across loops), sorted by block index.
        blocks = sorted(
            (lbl for lbl in prof.records if lbl.startswith("block[")),
            key=lambda s: s,
        )
        if blocks:
            print("  per-PairUpdateBlock mean time (ms), averaged over loops:")
            for lbl in blocks:
                rec = prof.records[lbl]
                mean_ms = rec.total_time / rec.count * 1e3 if rec.count else 0.0
                print(f"    {lbl:<12}{mean_ms:8.3f}  (n={rec.count})")

    msa = prof.records.get(MSA_ENCODER)
    if msa and msa.count:
        print(f"msa_encoder per loop iteration: {_summarize(msa.times)}")

    print("\n=== Step 2 (deep): diffusion trunk ===")
    step = prof.records.get(DIFFUSION_STEP)
    if step and step.count:
        print(f"diffusion_module per sampling step: {_summarize(step.times)}")
        if step.peaks:
            print(
                f"  per-step peak GPU: max {_fmt_gb(max(step.peaks)).strip()} GB  "
                f"(Δ over baseline {max(0, max(step.peaks) - baseline_bytes) / _GB:.2f} GB)"
            )
        # Flag a slow first step (common: graph/alloc warm-up inside the loop).
        if step.count > 1:
            first = step.times[0] * 1e3
            rest = statistics.mean(step.times[1:]) * 1e3
            if first > 1.5 * rest:
                print(
                    f"  note: first step {first:.2f}ms vs rest mean {rest:.2f}ms "
                    "(warm-up outlier)"
                )


def print_offload_summary(
    offloader: "LMOffloader", baseline_bytes: int, passes: int
) -> None:
    print("\n=== ESM-C CPU offload (--offload-lm) ===")
    if not offloader.active:
        print("offload inactive (no _esmc or non-CUDA device).")
        return
    parked = offloader.resident_parked
    print(f"resident GPU, LM on GPU (baseline): {baseline_bytes / _GB:.2f} GB")
    if parked is not None:
        print(
            f"resident GPU, LM parked on CPU:     {parked / _GB:.2f} GB  "
            f"(freed {(baseline_bytes - parked) / _GB:.2f} GB during the trunk)"
        )
    n = max(1, passes)
    print(
        f"transfer cost/pass: ->GPU {offloader.to_gpu_time / n * 1e3:.1f} ms, "
        f"->CPU {offloader.to_cpu_time / n * 1e3:.1f} ms "
        f"(total {(offloader.to_gpu_time + offloader.to_cpu_time) / n * 1e3:.1f} ms "
        f"over {offloader.moves} move(s))"
    )
    print(
        "Note: with offload on, the module 'peak(GB)' column reflects the lean "
        "footprint; compare it against the no-offload run's peaks."
    )


def build_module_json(
    prof: ModuleProfiler,
    module_label: str,
    block_prefix: str,
    config: dict,
    baseline_bytes: int,
) -> dict:
    """Per-module profile: module-level time slices + per-iteration peak memory,
    plus per-block time slices (blocks are nested, so time-only)."""
    rec = prof.records.get(module_label)
    module_block = None
    if rec is not None and rec.count:
        module_block = {
            "calls": rec.count,
            "total_ms": round(rec.total_time * 1e3, 3),
            "mean_ms": round(rec.total_time / rec.count * 1e3, 3),
            "per_call_ms": [round(t * 1e3, 3) for t in rec.times],
            "per_call_peak_gb": [round(p / _GB, 4) for p in rec.peaks],
            "per_call_delta_peak_gb": [
                round(max(0, p - baseline_bytes) / _GB, 4) for p in rec.peaks
            ],
        }
    blocks = {}
    for label in sorted(lbl for lbl in prof.records if lbl.startswith(block_prefix)):
        r = prof.records[label]
        if not r.count:
            continue
        blocks[label] = {
            "calls": r.count,
            "total_ms": round(r.total_time * 1e3, 3),
            "mean_ms": round(r.total_time / r.count * 1e3, 3),
            "per_call_ms": [round(t * 1e3, 3) for t in r.times],
        }
    return {
        "config": config,
        "baseline_resident_gb": round(baseline_bytes / _GB, 4),
        "module": module_block,
        "blocks": blocks,
        "notes": (
            "per_call_* has one entry per recycling iteration (module call). "
            "Block per_call_ms is time per call; blocks are nested under the "
            "module so per-block memory is not separately attributable. "
            "peak_gb is device-global allocated during that module call."
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_csv_rows(
    prof: ModuleProfiler, input_name: str, baseline_bytes: int
) -> list[dict]:
    rows = []
    for label, rec in prof.records.items():
        if rec.count == 0:
            continue
        peak = rec.peak
        rows.append(
            {
                "input": input_name,
                "module": label,
                "calls": rec.count,
                "total_ms": round(rec.total_time * 1e3, 3),
                "mean_ms": round(rec.total_time / rec.count * 1e3, 3),
                "peak_gb": round(peak / _GB, 4) if peak else "",
                "delta_peak_gb": round(max(0, peak - baseline_bytes) / _GB, 4)
                if peak
                else "",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_fold(builder, model, spec, args, seed):
    """One full fold() pass with the run_esmfold2 default arguments."""
    msa_max_depth = args.msa_max_depth if args.msa_max_depth > 0 else None
    return builder.fold(
        model,
        spec,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=1,
        seed=seed,
        msa_max_depth=msa_max_depth,
        complex_id="profile",
    )


def _structure_sanity(last_result) -> None:
    res = last_result[0] if isinstance(last_result, list) else last_result
    ptm = f"{res.ptm:.3f}" if getattr(res, "ptm", None) is not None else "n/a"
    if res is not None and getattr(res, "plddt", None) is not None:
        mean_plddt = f"{float(res.plddt.float().mean()):.3f}"
    else:
        mean_plddt = "n/a"
    print(f"\nStructure sanity: pTM={ptm}, mean pLDDT={mean_plddt}")


def run_torch_profiler(builder, model, spec, args, device, offloader) -> int:
    """Trace one fold pass with torch.profiler: GPU kernels (CUDA activities) +
    GPU memory (profile_memory) + record_function region labels. Writes a
    Perfetto-loadable chrome trace and a GPU memory timeline."""
    import torch.profiler as P

    labeler = RecordFunctionLabeler()
    install_record_labels(labeler, model)

    activities = [P.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(P.ProfilerActivity.CUDA)
    else:
        print("WARNING: non-CUDA device — no GPU kernels will be captured.")

    # Trace ONLY the ESMFold2 structure module (after ESM-C). Start the profiler
    # right after the LM stage (language_model post-hook, fallback folding_trunk
    # pre-hook) and stop at the end, so the ESM-C backbone + its offload CPU<->GPU
    # transfers (which run earlier in forward) are excluded.
    # record_shapes + with_stack are REQUIRED by export_memory_timeline.
    prof = P.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
    )
    state = {"started": False}
    want_snapshot = args.tensorboard is not None or args.mem_snapshot

    def _start_once(*_):
        if not state["started"]:
            state["started"] = True
            if want_snapshot and device.type == "cuda":
                # Record allocator block history (after ESM-C) for the snapshot
                # viewer — shows the allocated-block status at the peak.
                torch.cuda.memory._record_memory_history(max_entries=200_000)
            prof.start()

    trig_handles = []
    lm_shim = getattr(model, "language_model", None)
    if lm_shim is not None:
        trig_handles.append(lm_shim.register_forward_hook(lambda m, a, o: _start_once()))
    trunk = getattr(model, "folding_trunk", None)
    if trunk is not None:
        trig_handles.append(trunk.register_forward_pre_hook(lambda m, a: _start_once()))

    print(
        "Tracing the ESMFold2 structure module only (profiler starts after ESM-C)..."
    )
    last_result = None
    try:
        last_result = run_fold(builder, model, spec, args, args.seed)
    finally:
        if state["started"]:
            prof.stop()
        else:
            print("WARNING: profiler never started (no LM/trunk hook fired).")
        for h in trig_handles:
            h.remove()
        labeler.remove()
        if offloader is not None:
            offloader.remove()

    _structure_sanity(last_result)

    # TensorBoard export (torch_tb_profiler reads this: Trace + Memory views).
    if args.tensorboard is not None:
        os.makedirs(args.tensorboard, exist_ok=True)
        P.tensorboard_trace_handler(args.tensorboard)(prof)
        print(f"Wrote TensorBoard profile to {args.tensorboard}/ "
              f"(view: tensorboard --logdir {args.tensorboard})")

    # Memory snapshot: allocator block-level state over the traced window — open
    # the .pickle at https://pytorch.org/memory_viz and scrub to the peak to see
    # the allocated blocks at peak time.
    if want_snapshot and device.type == "cuda":
        stem = Path(args.input).stem
        snap = (
            os.path.join(args.tensorboard, f"{stem}_memory_snapshot.pickle")
            if args.tensorboard is not None
            else f"{args.torch_trace}_memory_snapshot.pickle"
        )
        try:
            torch.cuda.memory._dump_snapshot(snap)
            print(f"Wrote CUDA memory snapshot to {snap} (open at pytorch.org/memory_viz)")
        except Exception as exc:
            print(f"WARNING: memory snapshot dump failed: {exc!r}")
        torch.cuda.memory._record_memory_history(enabled=None)

    if args.torch_trace is not None:
        trace_path = f"{args.torch_trace}.pt.trace.json.gz"
        prof.export_chrome_trace(trace_path)
        print(f"Wrote GPU+CPU chrome trace to {trace_path} (Perfetto / chrome://tracing)")

        if device.type == "cuda":
            for ext in ("_mem_timeline.html", "_mem_timeline.json"):
                mem_path = f"{args.torch_trace}{ext}"
                try:
                    prof.export_memory_timeline(mem_path, device="cuda:0")
                    print(f"Wrote GPU memory timeline to {mem_path}")
                except Exception as exc:  # pragma: no cover
                    print(f"WARNING: memory-timeline export to {mem_path} failed: {exc!r}")

    # Quick top-kernel summary so the run is useful even before opening Perfetto.
    try:
        key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        print("\nTop ops by " + key + ":")
        print(prof.key_averages().table(sort_by=key, row_limit=12))
    except Exception as exc:
        print(f"(kernel summary unavailable: {exc!r})")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile ESMFold2 per-module time and peak GPU memory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Job JSON spec.")
    p.add_argument("--model-id", default="biohub/ESMFold2")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    p.add_argument("--num-loops", type=int, default=16)
    p.add_argument("--num-sampling-steps", type=int, default=200)
    p.add_argument("--msa-max-depth", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0, help="Fixed seed for parity.")
    p.add_argument(
        "--warmup", type=int, default=1, help="Discarded warm-up fold() passes."
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Measured fold() passes to average."
    )
    p.add_argument(
        "--deep",
        action="store_true",
        help="Step 2: per loop-iteration / per-block / per-step detail.",
    )
    p.add_argument(
        "--offload-lm",
        action="store_true",
        help="Park the ESM-C backbone (model._esmc) on CPU during the trunk/"
        "diffusion phases to free its ~12 GB; moved back to GPU for the LM step.",
    )
    p.add_argument(
        "--trunk-layers",
        type=int,
        default=None,
        help="Truncate folding_trunk to the first N PairUpdateBlocks (default: all "
        "48). Use a small N (e.g. 2) for a compact --deep per-block report.",
    )
    p.add_argument(
        "--msa-layers",
        type=int,
        default=None,
        help="Truncate msa_encoder to the first N MSAEncoderBlocks (default: all). "
        "Use a small N for a compact --deep per-block report.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="With --deep, write separate per-module JSON profiles: "
        "<prefix>_pair_trunk.json and <prefix>_msa_encoder.json (time slices + "
        "per-iteration peak memory).",
    )
    p.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Write a host-only Chrome Trace Event Format file (wall-clock of the "
        "hook regions). For GPU kernels/memory use --torch-trace instead.",
    )
    p.add_argument(
        "--torch-trace",
        type=str,
        default=None,
        help="Run one pass under torch.profiler (CPU+CUDA activities, "
        "profile_memory) and write <PREFIX>.pt.trace.json.gz (GPU kernels + memory "
        "counters; Perfetto/chrome://tracing) and <PREFIX>_mem_timeline.html "
        "(allocated GPU memory over time). Regions labelled via record_function. "
        "Use small --num-loops/--trunk-layers/--msa-layers/--num-sampling-steps.",
    )
    p.add_argument(
        "--kernel-backend",
        default="default",
        choices=["default", "none", "fused", "cuequivariance"],
        help="Trimul/kernel backend. 'default' leaves load_model's choice "
        "(cuequivariance if available else pure-PyTorch). 'fused' uses the vendored "
        "Triton kernels; 'none' forces pure-PyTorch.",
    )
    p.add_argument(
        "--apply-torch-compile",
        action="store_true",
        help="Call the model's built-in apply_torch_compile() (compiles "
        "PairUpdateBlock / MSAEncoderBlock / DiffusionModule / DiffusionTransformer "
        "module-by-module). Forces kernel backend to None — compile does not stack "
        "with the cueq/Triton custom ops.",
    )
    p.add_argument(
        "--compile-mode",
        default="fixed_seqlen",
        choices=["fixed_seqlen", "dynamic_seqlen"],
        help="apply_torch_compile mode: fixed_seqlen recompiles per sequence "
        "length; dynamic_seqlen compiles once for varying length.",
    )
    p.add_argument(
        "--tensorboard",
        type=str,
        default=None,
        help="Write a TensorBoard profile (torch_tb_profiler) to this logdir AND a "
        "CUDA memory snapshot (allocator block status, for pytorch.org/memory_viz). "
        "Profiles the structure module only (starts after ESM-C).",
    )
    p.add_argument(
        "--mem-snapshot",
        action="store_true",
        help="Record a CUDA allocator memory snapshot (block status incl. peak) "
        "even without --tensorboard.",
    )
    p.add_argument(
        "--mode",
        default=None,
        choices=["cueq", "compile", "hybrid", "cueq-msa"],
        help="Acceleration strategy via run_esmfold2.configure_acceleration. "
        "'cueq' = cueq kernels (MSA pure-PyTorch); 'compile' = full "
        "apply_torch_compile (no cueq); 'hybrid' = cueq + compiled msa_encoder; "
        "'cueq-msa' = cueq everywhere incl. the MSA trimul. Overrides "
        "--kernel-backend / --apply-torch-compile when set.",
    )
    p.add_argument(
        "--opm-chunk",
        type=int,
        default=64,
        help="MSA-encoder OuterProductMean chunk size (with --mode). 0 disables "
        "chunking (unchunked OPM — the big [L,L,c,d] transient).",
    )
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)

    job = load_job(args.input)
    total_res = sum(len(c["sequence"]) * c["copies"] for c in job["chains"])
    print(
        f"Job '{job['job_name']}': {len(job['chains'])} unique chain(s), "
        f"{total_res} residues. dtype={args.dtype} device={device} "
        f"mode={'deep' if args.deep else 'step1'}"
    )

    msa_max_depth = args.msa_max_depth if args.msa_max_depth > 0 else None
    model = load_model(args.model_id, args.device, _DTYPES[args.dtype])

    # Optional kernel-backend override (trimul: pure-PyTorch / Triton fused / cueq).
    if args.mode is None and args.kernel_backend != "default":
        backend = None if args.kernel_backend == "none" else args.kernel_backend
        model.set_kernel_backend(backend)
        print(f"Kernel backend set to {backend!r}.")

    # Optional trunk truncation: keep the first N PairUpdateBlocks (smaller --deep
    # report; structure output is meaningless but per-block timing is unaffected).
    if args.trunk_layers is not None:
        blocks = model.folding_trunk.blocks
        n = max(1, min(args.trunk_layers, len(blocks)))
        model.folding_trunk.blocks = blocks[:n]
        print(f"Truncated folding_trunk to {n}/{len(blocks)} PairUpdateBlocks.")

    if args.msa_layers is not None and getattr(model, "msa_encoder", None) is not None:
        mblocks = model.msa_encoder.blocks
        n = max(1, min(args.msa_layers, len(mblocks)))
        model.msa_encoder.blocks = mblocks[:n]
        print(f"Truncated msa_encoder to {n}/{len(mblocks)} MSAEncoderBlocks.")

    if args.mode is not None:
        from run_esmfold2 import configure_acceleration

        label = configure_acceleration(
            model,
            use_cueq=args.mode in ("cueq", "hybrid", "cueq-msa"),
            use_compile=args.mode in ("compile", "hybrid"),
            cueq_msa=args.mode == "cueq-msa",
            opm_chunk=(None if args.opm_chunk == 0 else args.opm_chunk),
        )
        print(f"mode={args.mode}: {label}")
    elif args.apply_torch_compile:
        if not hasattr(model, "apply_torch_compile"):
            print("WARNING: model has no apply_torch_compile(); skipping.")
        else:
            # Allow data-dependent output-shape ops (the atom unpadding produces
            # unbacked symints that otherwise fail Inductor).
            import torch._dynamo as _dynamo

            _dynamo.config.capture_scalar_outputs = True
            _dynamo.config.capture_dynamic_output_shape_ops = True
            # Built-in compile does not stack with cueq/Triton custom ops.
            model.set_kernel_backend(None)
            model.apply_torch_compile(mode=args.compile_mode)
            print(
                f"apply_torch_compile(mode={args.compile_mode}) applied; kernel "
                "backend forced to None (compile doesn't stack with cueq). First "
                "warm-up pass pays the compilation cost."
            )

    builder = ESMFold2InputBuilder()
    spec = build_input(job, msa_max_depth)

    # Baseline: resident memory after load (≈ model weights, dominated by ESMC-6B).
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
    else:
        baseline_bytes = 0
    print(f"Resident GPU memory after load: {baseline_bytes / _GB:.2f} GB")

    # Optional ESM-C CPU offload (installed before warm-up so the measured passes
    # see the steady-state CPU<->GPU move cost).
    offloader = None
    if args.offload_lm:
        offloader = install_lm_offload(model, device)
        if offloader.active:
            print("ESM-C CPU offload enabled (_esmc parked on CPU during trunk).")
        else:
            print("ESM-C CPU offload requested but inactive (no _esmc / non-CUDA).")

    # Warm-up (uninstrumented): excludes cuDNN autotune / lazy init / first alloc.
    for w in range(args.warmup):
        print(f"Warm-up pass {w + 1}/{args.warmup}...")
        run_fold(builder, model, spec, args, args.seed)
    if offloader is not None:
        offloader.reset()  # report transfer cost from measured passes only

    # torch.profiler path: GPU kernels + memory (separate from ModuleProfiler).
    if args.torch_trace is not None or args.tensorboard is not None or args.mem_snapshot:
        rc = run_torch_profiler(builder, model, spec, args, device, offloader)
        return rc

    # Install instrumentation and run measured passes.
    prof = ModuleProfiler(device)
    if args.deep:
        install_deep(prof, model)
    else:
        install_step1(prof, model)
    if args.trace_out is not None:
        prof.enable_trace()

    last_result = None
    try:
        for r in range(args.repeats):
            print(f"Measured pass {r + 1}/{args.repeats}...")
            last_result = run_fold(builder, model, spec, args, args.seed)
    finally:
        prof.remove()
        if offloader is not None:
            offloader.remove()

    # Structure sanity (confirms instrumentation didn't corrupt the run).
    res = last_result[0] if isinstance(last_result, list) else last_result
    ptm = f"{res.ptm:.3f}" if getattr(res, "ptm", None) is not None else "n/a"
    if res is not None and getattr(res, "plddt", None) is not None:
        mean_plddt = f"{float(res.plddt.float().mean()):.3f}"
    else:
        mean_plddt = "n/a"
    print(f"\nStructure sanity: pTM={ptm}, mean pLDDT={mean_plddt}")

    print_step1_table(prof, baseline_bytes)
    if args.deep:
        print_deep_report(prof, baseline_bytes)
    if offloader is not None:
        print_offload_summary(offloader, baseline_bytes, args.repeats)

    if args.csv is not None:
        rows = collect_csv_rows(prof, job["job_name"], baseline_bytes)
        write_csv(args.csv, rows)
        print(f"\nWrote per-module CSV to {args.csv}")

    if args.json_out is not None and args.deep:
        config = {
            "input": job["job_name"],
            "residues": total_res,
            "dtype": args.dtype,
            "device": str(device),
            "num_loops": args.num_loops,
            "recycling_iterations": args.num_loops + 1,
            "num_sampling_steps": args.num_sampling_steps,
            "trunk_layers": args.trunk_layers,
            "msa_layers": args.msa_layers,
            "offload_lm": bool(args.offload_lm),
            "kernel_backend": args.kernel_backend,
        }
        pair_path = Path(f"{args.json_out}_pair_trunk.json")
        msa_path = Path(f"{args.json_out}_msa_encoder.json")
        pair_path.write_text(
            json.dumps(
                build_module_json(prof, PAIR_TRUNK, "block[", config, baseline_bytes),
                indent=2,
            )
        )
        msa_path.write_text(
            json.dumps(
                build_module_json(
                    prof, MSA_ENCODER, "msa_block[", config, baseline_bytes
                ),
                indent=2,
            )
        )
        print(f"Wrote per-module JSON: {pair_path} and {msa_path}")

    if args.trace_out is not None:
        trace = {
            "traceEvents": prof.events,
            "displayTimeUnit": "ms",
            "metadata": {
                "input": job["job_name"],
                "dtype": args.dtype,
                "num_loops": args.num_loops,
                "trunk_layers": args.trunk_layers,
                "msa_layers": args.msa_layers,
                "offload_lm": bool(args.offload_lm),
            },
        }
        args.trace_out.write_text(json.dumps(trace))
        print(
            f"Wrote Chrome trace ({len(prof.events)} events) to {args.trace_out} "
            "— open in chrome://tracing or https://ui.perfetto.dev"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
