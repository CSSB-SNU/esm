#!/usr/bin/env python
"""NVTX-annotate one ESMFold2 fold for Nsight Systems (nsys) capture.

This is the *annotation driver* for nsys profiling of the ESMFold2 trunks (pair
trunk + diffusion trunk). It is intentionally separate from
``profile_esmfold2.py`` (which does host-side wall-clock timing + torch.profiler
traces): this script does no measuring of its own. It only

  1. lays down NVTX ranges around each unit — the pair trunk (``folding_trunk``)
     and every ``PairUpdateBlock``, the ``msa_encoder`` and every
     ``MSAEncoderBlock``, each diffusion step (``diffusion_module``), and the
     ESM-C backbone — via forward hooks (non-invasive; no edits to the installed
     ``transformers`` package), and
  2. brackets the *measured* fold in ``cudaProfilerStart/Stop`` so an
     ``nsys profile --capture-range=cudaProfilerApi`` run records only that fold,
     skipping model load and the warm-up pass.

Everything else — the CPU timeline, GPU timeline, HtoD ("h2g") copies, GPU
memory-operation time, and SM/GPU utilization — is collected by ``nsys`` itself.
Launch this through ``scripts/nsys_profile_esmfold2.sh`` (or a raw ``nsys
profile --capture-range=cudaProfilerApi ...``), not on its own.

It reuses the loading / input-building machinery from ``run_esmfold2.py``.

Example
-------
    # Normally invoked via the wrapper:
    srun --gres=gpu:1 --pty scripts/nsys_profile_esmfold2.sh examples/H2343.json

    # Equivalent raw invocation:
    srun --gres=gpu:1 --pty nsys profile --capture-range=cudaProfilerApi \
        --trace=cuda,nvtx,osrt,cublas,cudnn --cuda-memory-usage=true \
        --gpu-metrics-devices=cuda-visible \
        python scripts/nsys_profile_esmfold2.py \
            --input examples/H2343.json --dtype float32 \
            --num-loops 4 --num-sampling-steps 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/nsys_profile_esmfold2.py` without installing
# the package: put the repo root (parent of scripts/) on the import path, and
# this scripts/ dir so we can import the sibling run_esmfold2 helpers.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for _p in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: E402
from run_esmfold2 import (  # noqa: E402
    _DTYPES,
    build_input,
    configure_acceleration,
    install_lm_offload,
    load_job,
    load_model,
)


# --------------------------------------------------------------------------- #
# NVTX region labelling
# --------------------------------------------------------------------------- #


class NVTXLabeler:
    """Pushes ``torch.cuda.nvtx`` ranges around modules via forward hooks.

    nsys captures NVTX ranges natively and overlays them on the CPU/GPU timeline,
    correlating each range to the CUDA kernels + memcpys it launches. Stack-based
    push/pop → nested modules nest correctly. Non-invasive: forward hooks only.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._depth = 0

    def wrap(self, module: torch.nn.Module | None, label: str) -> None:
        if module is None:
            return

        def pre_hook(_m, _args):
            torch.cuda.nvtx.range_push(label)
            self._depth += 1

        def post_hook(_m, _args, _out):
            if self._depth > 0:
                torch.cuda.nvtx.range_pop()
                self._depth -= 1

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def remove(self) -> None:
        # Close any still-open ranges (e.g. on error), then detach hooks.
        while self._depth > 0:
            torch.cuda.nvtx.range_pop()
            self._depth -= 1
        for h in self._handles:
            h.remove()
        self._handles.clear()


def install_nvtx_labels(labeler: NVTXLabeler, model: torch.nn.Module) -> None:
    """Wrap the profiled units in NVTX ranges: ESM-C, pair trunk + blocks, MSA
    encoder + blocks, and each diffusion step."""
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

    labeler.wrap(
        getattr(model.structure_head, "diffusion_module", None), "diffusion_step"
    )


# --------------------------------------------------------------------------- #
# Fold driver
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
        complex_id="nsys",
    )


def _structure_sanity(last_result) -> None:
    res = last_result[0] if isinstance(last_result, list) else last_result
    ptm = f"{res.ptm:.3f}" if getattr(res, "ptm", None) is not None else "n/a"
    if res is not None and getattr(res, "plddt", None) is not None:
        mean_plddt = f"{float(res.plddt.float().mean()):.3f}"
    else:
        mean_plddt = "n/a"
    print(f"\nStructure sanity: pTM={ptm}, mean pLDDT={mean_plddt}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NVTX-annotate one ESMFold2 fold for Nsight Systems capture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Job JSON spec.")
    p.add_argument("--model-id", default="biohub/ESMFold2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=list(_DTYPES),
        help="Weight-storage dtype (float32 only). ESMFold2 handles mixed "
        "precision internally via autocast (bf16 trunk, fp32 diffusion/confidence "
        "head), so the profiled run already reflects the paper's precision; a "
        "blanket bf16/fp16 cast is unsupported (breaks the fp32 submodules).",
    )
    p.add_argument(
        "--num-loops",
        type=int,
        default=4,
        help="Pair-trunk recycling iterations. Keep small for nsys (the trunk "
        "repeats identical work per iteration; a full 16 makes a huge .nsys-rep).",
    )
    p.add_argument(
        "--num-sampling-steps",
        type=int,
        default=20,
        help="Diffusion sampling steps. Keep small for nsys (each step is "
        "identical work; a full 200 makes a huge .nsys-rep).",
    )
    p.add_argument("--msa-max-depth", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Discarded warm-up fold() passes (run BEFORE cudaProfilerStart, so "
        "they are excluded from the nsys capture).",
    )
    p.add_argument(
        "--offload-lm",
        action="store_true",
        help="Park the ESM-C backbone on CPU during trunk/diffusion (frees its "
        "GPU memory; adds a CPU<->GPU transfer visible as HtoD/DtoH in the trace).",
    )
    p.add_argument(
        "--trunk-layers",
        type=int,
        default=None,
        help="Truncate folding_trunk to the first N PairUpdateBlocks (compact "
        "trace; structure output is meaningless but per-block NVTX is unaffected).",
    )
    p.add_argument(
        "--msa-layers",
        type=int,
        default=None,
        help="Truncate msa_encoder to the first N MSAEncoderBlocks.",
    )
    p.add_argument(
        "--mode",
        default="cueq-msa",
        choices=["cueq", "compile", "hybrid", "cueq-msa", "none"],
        help="Acceleration strategy via run_esmfold2.configure_acceleration. "
        "'cueq' = cueq kernels; 'cueq-msa' = cueq incl. MSA trimul; 'compile' = "
        "full apply_torch_compile; 'hybrid' = cueq + compiled msa_encoder; "
        "'none' = pure-PyTorch.",
    )
    p.add_argument(
        "--opm-chunk",
        type=int,
        default=64,
        help="MSA-encoder OuterProductMean chunk size. 0 disables chunking.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        print("ERROR: nsys profiling requires a CUDA device (got "
              f"{device}). Run under srun --gres=gpu:1.")
        return 1

    job = load_job(args.input)
    total_res = sum(len(c["sequence"]) * c["copies"] for c in job["chains"])
    print(
        f"Job '{job['job_name']}': {len(job['chains'])} unique chain(s), "
        f"{total_res} residues. dtype={args.dtype} device={device} "
        f"loops={args.num_loops} steps={args.num_sampling_steps}"
    )

    msa_max_depth = args.msa_max_depth if args.msa_max_depth > 0 else None
    model = load_model(args.model_id, args.device, _DTYPES[args.dtype])

    # Optional block truncation for a compact trace.
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

    # Acceleration strategy. Default 'cueq-msa' mirrors run_esmfold2.py's working
    # config (cueq_msa=not --no-cueq): cueq kernels EVERYWHERE incl. MSA trimul.
    label = configure_acceleration(
        model,
        use_cueq=args.mode in ("cueq", "hybrid", "cueq-msa"),
        use_compile=args.mode in ("compile", "hybrid"),
        cueq_msa=args.mode == "cueq-msa",
        opm_chunk=(None if args.opm_chunk == 0 else args.opm_chunk),
    )
    print(f"Acceleration: {label}")

    builder = ESMFold2InputBuilder()
    spec = build_input(job, msa_max_depth)

    offloader = None
    if args.offload_lm:
        offloader = install_lm_offload(model, device)
        print(
            "ESM-C CPU offload enabled."
            if offloader.active
            else "ESM-C CPU offload requested but inactive (no _esmc / non-CUDA)."
        )

    # Warm-up OUTSIDE the capture window (cuDNN autotune / lazy init / first
    # alloc / torch.compile). These run before cudaProfilerStart, so nsys —
    # launched with --capture-range=cudaProfilerApi — never records them.
    for w in range(args.warmup):
        print(f"Warm-up pass {w + 1}/{args.warmup} (not captured)...")
        run_fold(builder, model, spec, args, args.seed)

    labeler = NVTXLabeler()
    install_nvtx_labels(labeler, model)

    cudart = torch.cuda.cudart()
    print(
        "Capturing one NVTX-annotated fold between cudaProfilerStart/Stop. "
        "(If not launched under `nsys profile --capture-range=cudaProfilerApi`, "
        "these calls are no-ops and nothing is recorded.)"
    )
    last_result = None
    torch.cuda.synchronize()
    cudart.cudaProfilerStart()
    torch.cuda.nvtx.range_push("fold (full forward)")
    try:
        last_result = run_fold(builder, model, spec, args, args.seed)
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        cudart.cudaProfilerStop()
        labeler.remove()
        if offloader is not None:
            offloader.remove()

    _structure_sanity(last_result)
    print(
        "\nCapture done. Summarize the .nsys-rep with:\n"
        "  nsys stats --report nvtx_pushpop_sum,cuda_gpu_kern_sum,"
        "cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum <file>.nsys-rep\n"
        "or open it in the Nsight Systems GUI (nsys-ui) for the CPU/GPU timeline "
        "+ GPU-utilization rows."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
