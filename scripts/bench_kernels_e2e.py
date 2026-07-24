#!/usr/bin/env python
"""End-to-end before/after A/B benchmark for ESMFold2 kernel backends.

Folds one job with each kernel backend in turn and reports how the whole-model
**inference time** (``model.forward`` wall clock) and **peak GPU memory** change
relative to the pure-PyTorch (``none``) baseline. Complements
``scripts/bench_kernels.py`` (per-op micro-bench with CUDA graphs): this measures
the realistic full pipeline (ESM-C is included but constant across backends, so
the delta isolates the folding-trunk / diffusion kernels).

Accuracy is verified as a PASS/FAIL gate: with a fixed seed the noise draws are
identical across backends, so an accelerated backend must reproduce the
pure-PyTorch fold's confidence outputs within tolerance — max |ΔpLDDT|, |ΔpTM|,
|ΔipTM| and max |ΔPAE|. (These frame-invariant confidence signals are compared
rather than raw coordinates, which can differ by a rigid transform.) A kernel is
never reported as a win unless the fold it produces is numerically equivalent.

GPU compute node only (use srun), never the login node:

    srun -p gpu --gres=gpu:A6000:1 --mem=96G --cpus-per-task=8 \
      .venv/bin/python scripts/bench_kernels_e2e.py \
        --input examples/job.json --backends none,fused --dtype float32 \
        --num-sampling-steps 20 --num-loops 2 --csv bench_e2e_a6000.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from run_esmfold2 import (  # noqa: E402
    _DTYPES,
    _set_msa_opm_chunk,
    build_input,
    install_lm_offload,
    load_job,
    load_model,
)

try:
    from transformers.models.esmfold2.modeling_esmfold2_common import (
        CUE_AVAILABLE,
        TRITON_KERNELS_AVAILABLE,
    )
except ImportError:  # pragma: no cover
    CUE_AVAILABLE = False
    TRITON_KERNELS_AVAILABLE = False

_GB = 1024.0**3


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #


class ForwardTimer:
    """Wrap model.forward to accumulate sync-bracketed wall time per call."""

    def __init__(self, model, cuda: bool) -> None:
        self.model = model
        self.cuda = cuda
        self.times: list[float] = []
        self._orig = model.forward

        def wrapped(*a, **k):
            if self.cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                return self._orig(*a, **k)
            finally:
                if self.cuda:
                    torch.cuda.synchronize()
                self.times.append(time.perf_counter() - t0)

        model.forward = wrapped

    def remove(self) -> None:
        self.model.forward = self._orig

    def reset(self) -> None:
        self.times.clear()


def backend_available(name: str) -> tuple[bool, str]:
    if name == "none":
        return True, ""
    if name == "fused":
        return TRITON_KERNELS_AVAILABLE, "" if TRITON_KERNELS_AVAILABLE else \
            "vendored triton kernels not importable"
    if name == "cuequivariance":
        return CUE_AVAILABLE, "" if CUE_AVAILABLE else "cuequivariance not importable"
    raise ValueError(f"unknown backend {name!r}")


def apply_backend(model, name: str, opm_chunk: int | None) -> None:
    """Apply a built-in kernel backend for the A/B. OPM chunking is held fixed
    across backends so the delta isolates the kernel, not the memory tiling."""
    model.set_kernel_backend(None if name == "none" else name)
    if opm_chunk is not None:
        _set_msa_opm_chunk(model, opm_chunk)


# --------------------------------------------------------------------------- #
# Folding + measurement
# --------------------------------------------------------------------------- #


def run_fold(builder, model, spec, args, seed):
    msa_max_depth = args.msa_max_depth if args.msa_max_depth > 0 else None
    return builder.fold(
        model, spec,
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=1,
        seed=seed,
        msa_max_depth=msa_max_depth,
        complex_id="bench",
    )


def _as_result(res):
    return res[0] if isinstance(res, list) else res


@dataclass
class Row:
    backend: str
    fwd_ms: float
    delta_ms: float | None
    speedup: float | None
    peak_gb: float
    delta_peak_gb: float | None
    ptm: float | None
    d_ptm: float | None
    d_iptm: float | None
    max_d_plddt: float | None
    max_d_pae: float | None
    accuracy_verdict: str


def _scalar(x):
    return float(x) if x is not None else None


def confidence_deltas(res, ref) -> dict:
    d = {"d_ptm": None, "d_iptm": None, "max_d_plddt": None, "max_d_pae": None}
    if getattr(res, "ptm", None) is not None and getattr(ref, "ptm", None) is not None:
        d["d_ptm"] = abs(float(res.ptm) - float(ref.ptm))
    if getattr(res, "iptm", None) is not None and getattr(ref, "iptm", None) is not None:
        d["d_iptm"] = abs(float(res.iptm) - float(ref.iptm))
    if getattr(res, "plddt", None) is not None and getattr(ref, "plddt", None) is not None:
        d["max_d_plddt"] = float((res.plddt.float() - ref.plddt.float()).abs().max())
    if getattr(res, "pae", None) is not None and getattr(ref, "pae", None) is not None:
        d["max_d_pae"] = float((res.pae.float() - ref.pae.float()).abs().max())
    return d


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    cuda = device.type == "cuda"

    job = load_job(args.input)
    total_res = sum(len(c["sequence"]) * c["copies"] for c in job["chains"])
    print(f"Job '{job['job_name']}': {total_res} residues, dtype={args.dtype}, "
          f"device={device}, backends={args.backends}")
    print(f"(triton={TRITON_KERNELS_AVAILABLE}, cueq={CUE_AVAILABLE})")

    model = load_model(args.model_id, args.device, _DTYPES[args.dtype])
    builder = ESMFold2InputBuilder()
    msa_max_depth = args.msa_max_depth if args.msa_max_depth > 0 else None
    spec = build_input(job, msa_max_depth)

    if args.trunk_layers is not None:
        blocks = model.folding_trunk.blocks
        n = max(1, min(args.trunk_layers, len(blocks)))
        model.folding_trunk.blocks = blocks[:n]
        print(f"Truncated folding_trunk to {n}/{len(blocks)} blocks.")

    offloader = install_lm_offload(model, model.device) if args.offload_lm else None
    if offloader is not None and offloader.active:
        print("ESM-C CPU offload enabled.")

    timer = ForwardTimer(model, cuda)

    rows: list[Row] = []
    ref_result = None
    baseline_ms: float | None = None
    baseline_peak: float | None = None

    for name in args.backends.split(","):
        ok, reason = backend_available(name)
        if not ok:
            print(f"  [skip] backend={name}: {reason}")
            continue
        apply_backend(model, name, args.opm_chunk if args.opm_chunk > 0 else None)
        print(f"\n=== backend={name} ===")

        # Warm-up (uninstrumented) then measured passes.
        for _ in range(args.warmup):
            run_fold(builder, model, spec, args, args.seed)
        timer.reset()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        last = None
        for r in range(args.repeats):
            last = run_fold(builder, model, spec, args, args.seed)
        peak = torch.cuda.max_memory_allocated() / _GB if cuda else 0.0
        fwd_ms = (sum(timer.times) / max(1, len(timer.times))) * 1e3

        res = _as_result(last)
        if name == "none":
            ref_result = res
            baseline_ms = fwd_ms
            baseline_peak = peak
            deltas = {"d_ptm": 0.0, "d_iptm": 0.0, "max_d_plddt": 0.0, "max_d_pae": 0.0}
            verdict = "PASS"
        else:
            deltas = confidence_deltas(res, ref_result)
            verdict = "PASS"
            if deltas["d_ptm"] is not None and deltas["d_ptm"] > args.tol_ptm:
                verdict = "FAIL"
            if deltas["max_d_plddt"] is not None and deltas["max_d_plddt"] > args.tol_plddt:
                verdict = "FAIL"
            if deltas["max_d_pae"] is not None and deltas["max_d_pae"] > args.tol_pae:
                verdict = "FAIL"

        rows.append(Row(
            backend=name, fwd_ms=round(fwd_ms, 2),
            delta_ms=round(fwd_ms - baseline_ms, 2) if baseline_ms is not None else None,
            speedup=round(baseline_ms / fwd_ms, 3) if baseline_ms else None,
            peak_gb=round(peak, 4),
            delta_peak_gb=round(peak - baseline_peak, 4) if baseline_peak is not None else None,
            ptm=_scalar(getattr(res, "ptm", None)),
            d_ptm=deltas["d_ptm"], d_iptm=deltas["d_iptm"],
            max_d_plddt=deltas["max_d_plddt"], max_d_pae=deltas["max_d_pae"],
            accuracy_verdict=verdict,
        ))
        mp = (f"{float(res.plddt.float().mean()):.2f}"
              if getattr(res, "plddt", None) is not None else "n/a")
        print(f"  fwd={fwd_ms:.1f}ms peak={peak:.2f}GB pTM="
              f"{_scalar(getattr(res, 'ptm', None))} meanPLDDT={mp} verdict={verdict}")

    timer.remove()
    if offloader is not None:
        offloader.remove()

    print_table(rows)
    if args.csv is not None:
        write_csv(args.csv, rows)
        print(f"\nWrote {len(rows)} rows to {args.csv}")

    fails = [r for r in rows if r.accuracy_verdict == "FAIL"]
    if fails:
        print(f"\nACCURACY: {len(fails)} backend(s) FAILed vs the none-baseline fold.")
        if args.fail_on_accuracy:
            return 1
    else:
        print("\nACCURACY: all backends reproduce the baseline fold within tolerance.")
    return 0


# --------------------------------------------------------------------------- #
# Reporting / CLI
# --------------------------------------------------------------------------- #


def print_table(rows: list[Row]) -> None:
    hdr = (f"{'backend':<15}{'fwd(ms)':>10}{'Δms':>10}{'speedup':>9}"
           f"{'peak(GB)':>10}{'Δpeak':>9}{'pTM':>7}{'ΔpTM':>8}"
           f"{'maxΔpLDDT':>11}{'maxΔPAE':>9}{'acc':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        dms = f"{r.delta_ms:+.1f}" if r.delta_ms is not None else "-"
        sp = f"{r.speedup:.2f}x" if r.speedup else "-"
        dp = f"{r.delta_peak_gb:+.3f}" if r.delta_peak_gb is not None else "-"
        ptm = f"{r.ptm:.3f}" if r.ptm is not None else "n/a"
        dptm = f"{r.d_ptm:.4f}" if r.d_ptm is not None else "-"
        dpl = f"{r.max_d_plddt:.3f}" if r.max_d_plddt is not None else "-"
        dpae = f"{r.max_d_pae:.3f}" if r.max_d_pae is not None else "-"
        flag = "  <== FAIL" if r.accuracy_verdict == "FAIL" else ""
        print(f"{r.backend:<15}{r.fwd_ms:>10.1f}{dms:>10}{sp:>9}"
              f"{r.peak_gb:>10.3f}{dp:>9}{ptm:>7}{dptm:>8}"
              f"{dpl:>11}{dpae:>9}{r.accuracy_verdict:>6}{flag}")


def write_csv(path: Path, rows: list[Row]) -> None:
    fields = list(Row.__dataclass_fields__.keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end A/B kernel benchmark for ESMFold2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Job JSON spec.")
    p.add_argument("--model-id", default="biohub/ESMFold2")
    p.add_argument("--backends", default="none,fused",
                   help="Comma list of none,fused,cuequivariance.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES))
    p.add_argument("--num-loops", type=int, default=2)
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--msa-max-depth", type=int, default=1024)
    p.add_argument("--trunk-layers", type=int, default=None,
                   help="Truncate folding_trunk to N blocks (faster A/B).")
    p.add_argument("--opm-chunk", type=int, default=64,
                   help="MSA OuterProductMean chunk (held fixed across backends; 0=off).")
    p.add_argument("--offload-lm", action="store_true")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="Fixed seed (deterministic A/B).")
    p.add_argument("--tol-ptm", type=float, default=0.02)
    p.add_argument("--tol-plddt", type=float, default=1.0)
    p.add_argument("--tol-pae", type=float, default=1.0)
    p.add_argument("--fail-on-accuracy", action="store_true", default=True)
    p.add_argument("--no-fail-on-accuracy", dest="fail_on_accuracy", action="store_false")
    p.add_argument("--csv", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
