#!/usr/bin/env python
"""Capture a torch memory-history snapshot of ONE padded fold, with a
configurable ring-buffer size.

Exists because the diffab driver hardcoded max_entries=400_000, which the
512-token fold fits (274k events) but 1024/2048 folds overflow — the ring
keeps only the LAST 400k events, silently dropping the region that contains
the real peak, so peak reports built from those snapshots were wrong
(e.g. 2048 "peak" 31-36 GB vs the measured 65 GB).

Imports the measurement plumbing from the MAIN checkout's scripts dir (not a
copy), so PadTokens / backend patching / model loading behave exactly like
every previous diffab run.

Usage:
  python memviz_capture.py --input JOB --bucket N --attn fp32|bf16-triton|bf16-cuda \
      --out SNAP.pickle [--max-entries 3000000] [--num-loops 2] [--steps 50] \
      [--offload-lm] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_MAIN = Path("/home/mjkang/optimization/esm")
for p in (str(_MAIN), str(_MAIN / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from run_esmfold2 import (  # noqa: E402
    _DTYPES, build_input, configure_acceleration, install_lm_offload,
    load_job, load_model,
)
from graph_infer import PadTokens  # noqa: E402
from bench_kernels_miniworld import apply_miniworld  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--bucket", required=True, type=int)
    ap.add_argument("--attn", required=True,
                    choices=["fp32", "bf16-triton", "bf16-cuda"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-entries", type=int, default=3_000_000)
    ap.add_argument("--num-loops", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--msa-max-depth", type=int, default=1024)
    ap.add_argument("--offload-lm", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    job = load_job(args.input)
    total = sum(len(c["sequence"]) * c["copies"] for c in job["chains"])
    print(f"[job] {job['job_name']}: {total} res -> bucket {args.bucket}, "
          f"attn={args.attn}, max_entries={args.max_entries}")

    model = load_model("biohub/ESMFold2", "cuda", _DTYPES["float32"])
    configure_acceleration(model, use_cueq=False, use_compile=False,
                           cueq_msa=False, opm_chunk=64)
    if args.attn != "fp32":
        import os
        os.environ["ESM_ATTN_BIAS_IMPL"] = \
            "triton" if args.attn == "bf16-triton" else "cuda"
        revert = apply_miniworld(model, only={"attn_pair_bias"})
        print(f"[attn] patched {getattr(revert, 'counts', {})}")

    builder = ESMFold2InputBuilder()
    spec = build_input(job, args.msa_max_depth or None)
    model.forward = PadTokens(model.forward, buckets=[args.bucket])
    if args.offload_lm:
        install_lm_offload(model, model.device)
        print("[offload] ESM-C CPU offload enabled")

    def fold():
        return builder.fold(model, spec, num_loops=args.num_loops,
                            num_sampling_steps=args.steps,
                            num_diffusion_samples=1, seed=args.seed,
                            msa_max_depth=args.msa_max_depth or None,
                            complex_id="memviz")

    print("[warmup] fold ...")
    fold()
    torch.cuda.synchronize()
    torch.cuda.memory._record_memory_history(max_entries=args.max_entries)
    t0 = time.perf_counter()
    fold()
    torch.cuda.synchronize()
    print(f"[measured] fold {time.perf_counter()-t0:.1f}s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(args.out))
    torch.cuda.memory._record_memory_history(enabled=None)

    import pickle
    n = len(pickle.load(open(args.out, "rb"))["device_traces"][0])
    wrapped = "RING WRAPPED — INCREASE --max-entries!" \
        if n >= args.max_entries else "ok (no wrap)"
    print(f"[snapshot] {args.out}: {n} events / budget {args.max_entries} -> {wrapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
