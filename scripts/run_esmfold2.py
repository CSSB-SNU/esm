#!/usr/bin/env python
"""Fold a protein with ESMFold2 from local HuggingFace weights.

Reads a JSON job spec (sequence, MSA path, job name, output path), folds the
sequence with the local ESMFold2 model, and writes the predicted structure plus
confidence outputs (PAE, pTM/ipTM, pLDDT).

Example
-------
    srun --gres=gpu:1 --pty python scripts/run_esmfold2.py --input job.json

Job JSON (keys are matched case-insensitively, so the field names below or
`Sequence` / `MSApath` / `job name` / `output path` all work)::

    {
      "job_name": "ubiquitin",
      "sequence": "MQIFVKTLTGK...",
      "msa_path": "/path/to/ubiquitin.a3m",   # optional; fold without MSA if absent
      "output_path": "/path/to/outdir"
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Allow running as `python scripts/run_esmfold2.py` without installing the
# package: put the repo root (parent of scripts/) on the import path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.msa import MSA

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


# ----------------------------- job spec parsing ----------------------------- #


_KEY_ALIASES = {
    "jobname": "job_name",
    "job_name": "job_name",
    "sequence": "sequence",
    "seq": "sequence",
    "msapath": "msa_path",
    "msa_path": "msa_path",
    "msa": "msa_path",
    "outputpath": "output_path",
    "output_path": "output_path",
    "output": "output_path",
    "outdir": "output_path",
    "chains": "chains",
    "subunits": "chains",
    "copies": "copies",
    "copy": "copies",
    "count": "copies",
    "n": "copies",
    "stoich": "copies",
    "stoichiometry": "copies",
}


def _normalize_keys(raw: dict) -> dict:
    """Map case/space/underscore-insensitive JSON keys to canonical names."""
    out: dict = {}
    for key, value in raw.items():
        norm = key.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        canonical = _KEY_ALIASES.get(norm)
        if canonical is not None:
            out[canonical] = value
    return out


def validate_sequence(sequence: str) -> str:
    """Validate and normalize a protein sequence (uppercase, canonical AAs)."""
    sequence = "".join(sequence.split()).upper()
    if not sequence:
        raise ValueError("Sequence is empty.")
    invalid = sorted(set(sequence) - VALID_AAS)
    if invalid:
        raise ValueError(f"Invalid amino acid characters found: {invalid}")
    return sequence


def _parse_chain(raw: dict) -> dict:
    """Normalize one chain entry: {sequence, copies, msa_path}."""
    chain = _normalize_keys(raw)
    if "sequence" not in chain:
        raise ValueError(f"Chain entry is missing a sequence: {sorted(chain)}")
    copies = int(chain.get("copies", 1))
    if copies < 1:
        raise ValueError(f"Chain 'copies' must be >= 1, got {copies}.")
    return {
        "sequence": validate_sequence(chain["sequence"]),
        "copies": copies,
        "msa_path": chain.get("msa_path") or None,
    }


def load_job(input_path: Path) -> dict:
    """Parse a job spec into {job_name, output_path, chains: [{sequence, copies, msa_path}]}.

    Supports two shapes:
      * single chain — top-level ``sequence`` (+ optional ``msa_path``).
      * complex — a ``chains`` list, each with ``sequence``, ``copies``, ``msa_path``.
    """
    raw = json.loads(input_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Job spec must be a JSON object, got {type(raw).__name__}.")
    job = _normalize_keys(raw)

    missing = [k for k in ("job_name", "output_path") if k not in job]
    if missing:
        raise ValueError(
            f"Job spec is missing required field(s): {missing}. "
            f"Found keys: {sorted(job)}"
        )

    if "chains" in job:
        chains_raw = job["chains"]
        if not isinstance(chains_raw, list) or not chains_raw:
            raise ValueError("'chains' must be a non-empty list of chain entries.")
        chains = [_parse_chain(c) for c in chains_raw]
    elif "sequence" in job:
        chains = [
            {
                "sequence": validate_sequence(job["sequence"]),
                "copies": 1,
                "msa_path": job.get("msa_path") or None,
            }
        ]
    else:
        raise ValueError(
            "Job spec must contain either a top-level 'sequence' or a 'chains' list."
        )

    return {
        "job_name": job["job_name"],
        "output_path": job["output_path"],
        "chains": chains,
    }


# ------------------------------- model loading ------------------------------ #


_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _enable_cueq_ops() -> None:
    """Make cuequivariance's bundled ``libcue_ops.so`` loadable without requiring
    the user to set ``LD_LIBRARY_PATH``.

    The ``cuequivariance_ops_torch`` C++ extension links ``libcue_ops.so`` but
    does not RPATH it, and that lib in turn needs CUDA libs (``libnvrtc``/
    ``libcuda``) which ``torch`` loads into the process once imported. Preloading
    ``libcue_ops.so`` with ``RTLD_GLOBAL`` after torch is imported resolves the
    lazy ops import used by the ``cuequivariance`` kernel backend (trimul +
    attention_pair_bias). No-op if cuequivariance isn't installed / non-CUDA.
    """
    try:
        import ctypes
        import os.path as _osp

        import cuequivariance_ops  # type: ignore[import]

        lib = _osp.join(
            _osp.dirname(cuequivariance_ops.__file__), "lib", "libcue_ops.so"
        )
        if _osp.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass  # not installed / CPU-only — backend selection will fall back


def load_model(model_id: str, device: str, dtype: torch.dtype):
    """Load ESMFold2 (``ESMFold2Model``) from HuggingFace weights.

    ``from_pretrained`` loads the ESMC backbone itself (``load_esmc=True``), so we
    don't call ``load_esmc`` again.

    When cuequivariance is installed, the trimul/attention kernels are selected
    (``CUE_AVAILABLE``) — ~7-8x faster pair-trunk trimul. ``_enable_cueq_ops()``
    makes those kernels loadable. TF32 is enabled for the fp32 matmul path
    (diffusion module), matching the ESMFold2 paper's inference config.
    """
    # Match the paper's inference precision: TF32 on the fp32 matmul path.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _enable_cueq_ops()
    try:
        from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        from transformers.models.esmfold2.modeling_esmfold2_common import (
            CUE_AVAILABLE,
        )
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Could not import ESMFold2 from `transformers`. Install a transformers "
            "build that ships `transformers.models.esmfold2` (see the esm install "
            "instructions / binder_design tutorial)."
        ) from exc

    print(f"Loading {model_id} in {dtype} (this pulls the ESMC backbone on first run)...")
    # from_pretrained dispatches by config.type and loads ESMC internally.
    model = ESMFold2Model.from_pretrained(model_id, torch_dtype=dtype)
    model.set_kernel_backend("cuequivariance" if CUE_AVAILABLE else None)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model


def _set_msa_trimul_backend(model, backend: str) -> int:
    """Set the kernel backend on the MSA encoder's trimul blocks (tri_mul_out/in).

    model.set_kernel_backend() does NOT reach msa_encoder, and MSAEncoder has no
    set_kernel_backend — but its blocks hold real TriangleMultiplicativeUpdate
    instances, so we flip their backend directly. Returns #trimuls switched.
    """
    msa = getattr(model, "msa_encoder", None)
    n = 0
    if msa is not None and hasattr(msa, "blocks"):
        for blk in msa.blocks:
            for attr in ("tri_mul_out", "tri_mul_in"):
                t = getattr(blk, attr, None)
                if t is not None and hasattr(t, "set_kernel_backend"):
                    t.set_kernel_backend(backend)
                    n += 1
    return n


def _set_msa_opm_chunk(model, chunk: int | None) -> int:
    """Enable chunking on the MSA encoder's OuterProductMean (default-unwired).

    OPM forms a [B,L,L,d_hidden,d_hidden] outer product — O(L²·d_hidden²), the
    model's largest transient. The chunked path tiles it along the i-axis. The
    model never wires set_chunk_size into msa_encoder, so OPM defaults to
    unchunked; we set it per block. Returns #OPMs set.
    """
    msa = getattr(model, "msa_encoder", None)
    n = 0
    if msa is not None and hasattr(msa, "blocks"):
        for blk in msa.blocks:
            opm = getattr(blk, "outer_product_mean", None)
            if opm is not None and hasattr(opm, "set_chunk_size"):
                opm.set_chunk_size(chunk)
                n += 1
    return n


def configure_acceleration(
    model,
    use_cueq: bool,
    use_compile: bool,
    cueq_msa: bool = False,
    opm_chunk: int | None = 64,
) -> str:
    """Apply the kernel-backend + torch.compile strategy; returns a label.

    - cueq only        : cuequivariance kernels (trunk/diffusion/confidence trimul
                         + attention_pair_bias); MSA encoder stays pure-PyTorch.
    - compile only      : backend None + apply_torch_compile() (all blocks).
    - hybrid (cueq+compile): cueq everywhere it reaches AND torch.compile the MSA
                         encoder (the module cueq's propagation misses).
    - cueq_msa          : cueq everywhere INCLUDING the MSA encoder's trimul
                         (tri_mul_out/in), wired directly — no compile.
    """
    from transformers.models.esmfold2.modeling_esmfold2_common import CUE_AVAILABLE

    cueq_backend = "cuequivariance" if (use_cueq and CUE_AVAILABLE) else None

    # MSA-encoder OuterProductMean chunking (always-beneficial memory fix; the
    # model leaves it unwired so OPM defaults to unchunked). Independent of cueq.
    n_opm = _set_msa_opm_chunk(model, opm_chunk) if opm_chunk is not None else 0
    opm = f" + OPM chunk={opm_chunk} x{n_opm}" if n_opm else ""

    if use_compile and use_cueq:  # compile-MSA HYBRID
        model.set_kernel_backend(cueq_backend)
        msa = getattr(model, "msa_encoder", None)
        if msa is not None and hasattr(msa, "blocks"):
            import torch._dynamo as _dyn

            _dyn.config.capture_scalar_outputs = True
            _dyn.config.capture_dynamic_output_shape_ops = True
            for blk in msa.blocks:
                blk.forward = torch.compile(blk.forward, dynamic=True)
        return f"hybrid (cueq={cueq_backend} + compiled msa_encoder){opm}"

    if use_compile:  # FULL COMPILE (no cueq)
        model.set_kernel_backend(None)
        if hasattr(model, "apply_torch_compile"):
            model.apply_torch_compile()
        return f"compile (full apply_torch_compile, backend=None){opm}"

    model.set_kernel_backend(cueq_backend)  # CUEQ ONLY (or pure pytorch)
    if cueq_msa and cueq_backend is not None:
        n = _set_msa_trimul_backend(model, cueq_backend)
        return f"cueq-msa hybrid (cueq everywhere incl. {n} MSA trimuls){opm}"
    return (f"cueq ({cueq_backend})" if cueq_backend else "none (pure-pytorch)") + opm


# ------------------------------ ESM-C CPU offload --------------------------- #


class LMOffloader:
    """Parks ``model._esmc`` on CPU while the trunk/diffusion run.

    The ESM-C backbone (~24 GB in fp32 / ~12 GB in bf16) runs once at the start
    of ``forward`` and is never used by the folding trunk or the diffusion
    sampler, yet it stays resident on the GPU for the whole (dominant,
    memory-growing) folding phase. A forward-pre-hook moves it back to the GPU
    just before the LM forward; a forward-hook evicts it to CPU (+ ``empty_cache``)
    right after. Non-invasive — forward hooks only, no edits to the model code.

    Trade-off: each fold() pays a fixed CPU<->GPU transfer (~13 s for the fp32
    backbone), so only enable this when GPU memory is the constraint (long
    sequences, or the 24 GB A5000) — it is pure overhead on short folds.
    """

    def __init__(self, model, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.esmc = getattr(model, "_esmc", None)
        self.to_gpu_time = 0.0  # seconds, accumulated over LM forwards
        self.to_cpu_time = 0.0
        self.moves = 0
        self.resident_parked: int | None = None  # bytes allocated while parked
        self._handles: list = []

    @property
    def active(self) -> bool:
        return self.esmc is not None and self.device.type == "cuda"

    def _to_gpu(self) -> None:
        t0 = time.perf_counter()
        self.esmc.to(self.device)
        torch.cuda.synchronize()
        self.to_gpu_time += time.perf_counter() - t0

    def _to_cpu(self) -> None:
        t0 = time.perf_counter()
        self.esmc.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.resident_parked = torch.cuda.memory_allocated()
        self.to_cpu_time += time.perf_counter() - t0
        self.moves += 1

    def install(self) -> "LMOffloader":
        if not self.active:
            return self
        # Park immediately so the first trunk pass already runs lean.
        self.esmc.to("cpu")
        torch.cuda.empty_cache()
        self._handles.append(
            self.esmc.register_forward_pre_hook(lambda _m, _a: self._to_gpu())
        )
        self._handles.append(
            self.esmc.register_forward_hook(lambda _m, _a, _o: self._to_cpu())
        )
        return self

    def reset(self) -> None:
        self.to_gpu_time = self.to_cpu_time = 0.0
        self.moves = 0

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self.esmc is not None and self.device.type == "cuda":
            self.esmc.to(self.device)  # restore so the model is left as found


def install_lm_offload(model, device: torch.device) -> LMOffloader:
    """Install ESM-C CPU offload on ``model`` and return the handle."""
    return LMOffloader(model, device).install()


# --------------------------------- folding ---------------------------------- #


def _chain_ids(total: int) -> list[str]:
    """Generate `total` unique chain IDs: A..Z, then AA, AB, ... ."""
    import string

    letters = string.ascii_uppercase
    ids: list[str] = []
    i = 0
    while len(ids) < total:
        q, r = divmod(i, len(letters))
        ids.append((letters[q - 1] if q else "") + letters[r])
        i += 1
    return ids


def _load_chain_msa(chain: dict, msa_max_depth: int | None) -> MSA | None:
    if chain["msa_path"] is None:
        return None
    msa_path = Path(chain["msa_path"]).expanduser()
    if not msa_path.is_file():
        raise FileNotFoundError(f"MSA file not found: {msa_path}")
    # Cap the loaded depth; the model subsamples per-loop down to msa_max_depth.
    max_sequences = msa_max_depth if msa_max_depth and msa_max_depth > 0 else None
    msa = MSA.from_a3m(
        path=str(msa_path), remove_insertions=True, max_sequences=max_sequences
    )
    if msa.sequences and msa.sequences[0] != chain["sequence"]:
        warnings.warn(
            f"First MSA row in {msa_path.name} does not match the chain query "
            "sequence; make sure the MSA was built for this sequence.",
            stacklevel=2,
        )
    return msa


def build_input(job: dict, msa_max_depth: int | None) -> StructurePredictionInput:
    total_chains = sum(c["copies"] for c in job["chains"])
    all_ids = _chain_ids(total_chains)

    proteins: list[ProteinInput] = []
    cursor = 0
    for idx, chain in enumerate(job["chains"]):
        ids = all_ids[cursor : cursor + chain["copies"]]
        cursor += chain["copies"]
        msa = _load_chain_msa(chain, msa_max_depth)
        # ProteinInput.id is a single str for one copy, else a list of IDs.
        chain_id = ids[0] if len(ids) == 1 else ids
        proteins.append(
            ProteinInput(id=chain_id, sequence=chain["sequence"], msa=msa)
        )
        print(
            f"  chain group {idx + 1}: ids={ids}, len={len(chain['sequence'])}, "
            f"MSA={'depth=' + str(msa.depth) if msa is not None else 'none'}"
        )

    return StructurePredictionInput(sequences=proteins)


def write_outputs(
    result, job: dict, tag: str = "", seed: int | None = None, sample: int = 0
) -> tuple[list[Path], dict]:
    out_dir = Path(job["output_path"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['job_name']}{tag}"
    written: list[Path] = []

    # Structure (mmCIF; pLDDT is written into the B-factor column).
    cif_path = out_dir / f"{stem}.cif"
    cif_path.write_text(result.complex.to_mmcif())
    written.append(cif_path)

    # Per-residue pLDDT array.
    plddt = None
    if result.plddt is not None:
        plddt = result.plddt.detach().cpu().numpy()
        plddt_path = out_dir / f"{stem}_plddt.npy"
        np.save(plddt_path, plddt)
        written.append(plddt_path)

    # PAE matrix (only present when the confidence head ran).
    if result.pae is not None:
        pae_path = out_dir / f"{stem}_pae.npy"
        np.save(pae_path, result.pae.detach().cpu().numpy())
        written.append(pae_path)

    # Scalar confidence summary.
    confidence = {
        "job_name": job["job_name"],
        "seed": seed,
        "sample": sample,
        "ptm": float(result.ptm) if result.ptm is not None else None,
        "iptm": float(result.iptm) if result.iptm is not None else None,
        "mean_plddt": float(plddt.mean()) if plddt is not None else None,
        "num_residues": int(plddt.shape[0]) if plddt is not None else None,
    }
    conf_path = out_dir / f"{stem}_confidence.json"
    conf_path.write_text(json.dumps(confidence, indent=2))
    written.append(conf_path)

    return written, confidence


# ----------------------------------- CLI ------------------------------------ #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fold a protein with local ESMFold2 weights from a JSON job spec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Path to the job JSON spec."
    )
    parser.add_argument(
        "--model-id",
        default="biohub/ESMFold2",
        help="HuggingFace repo id for the ESMFold2 weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=list(_DTYPES),
        help="Compute/weights dtype. float32 is the reference path (fits a 48 GB "
        "GPU; ESMC-6B is ~24 GB). bfloat16/float16 halve memory but the blanket "
        "cast can hit dtype mismatches in the fp32 diffusion path; use with care.",
    )
    parser.add_argument("--num-loops", type=int, default=16)
    parser.add_argument("--num-sampling-steps", type=int, default=200)
    parser.add_argument(
        "--num-diffusion-samples",
        type=int,
        default=1,
        help="Diffusion samples drawn per seed (one trunk pass, N decoded structures).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base random seed. With --num-seeds N, runs seeds base..base+N-1. "
        "Omit for a single non-deterministic run (unless --num-seeds > 1, which "
        "then defaults the base to 0).",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of distinct seeds to fold (independent trunk passes). Total "
        "structures = num_seeds * num_diffusion_samples.",
    )
    parser.add_argument(
        "--msa-max-depth",
        type=int,
        default=1024,
        help="Max MSA rows kept/subsampled per loop. Set <=0 to use the full MSA.",
    )
    parser.add_argument(
        "--offload-lm",
        action="store_true",
        help="Park the ESM-C backbone on CPU during the trunk/diffusion phases to "
        "free its GPU memory (~24 GB fp32). Adds a fixed CPU<->GPU transfer per "
        "fold (~13 s fp32); use only when GPU memory is the constraint (long "
        "sequences / 24 GB GPU).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model. With cueq on (default) this is HYBRID mode "
        "(cueq kernels + compiled msa_encoder); with --no-cueq it is full "
        "apply_torch_compile. Requires torch ~2.8 (fails on 2.12).",
    )
    parser.add_argument(
        "--no-cueq",
        action="store_true",
        help="Disable cuequivariance kernels everywhere (trunk/diffusion/confidence "
        "AND the MSA-encoder trimul) — use pure-PyTorch / compile only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    job = load_job(args.input)
    total_chains = sum(c["copies"] for c in job["chains"])
    total_res = sum(len(c["sequence"]) * c["copies"] for c in job["chains"])
    print(
        f"Job '{job['job_name']}': {len(job['chains'])} unique chain(s), "
        f"{total_chains} chains total, {total_res} residues."
    )

    msa_max_depth = args.msa_max_depth if args.msa_max_depth > 0 else None

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")
    if args.num_seeds == 1:
        seeds: list[int | None] = [args.seed]
    else:
        base = args.seed if args.seed is not None else 0
        seeds = [base + i for i in range(args.num_seeds)]

    n_samples = args.num_diffusion_samples
    multi = args.num_seeds > 1 or n_samples > 1

    model = load_model(args.model_id, args.device, _DTYPES[args.dtype])
    # cueq covers trunk/diffusion/confidence AND the MSA-encoder trimul (combined):
    # cueq-MSA is strictly faster + lighter than leaving MSA on pure-PyTorch.
    accel = configure_acceleration(
        model,
        use_cueq=not args.no_cueq,
        use_compile=args.compile,
        cueq_msa=not args.no_cueq,
    )
    print(f"Acceleration: {accel}")
    if args.offload_lm:
        offloader = install_lm_offload(model, model.device)
        if offloader.active:
            print("ESM-C CPU offload enabled (_esmc parked on CPU during trunk).")
        else:
            print("ESM-C CPU offload requested but inactive (no _esmc / non-CUDA).")
    builder = ESMFold2InputBuilder()
    spec = build_input(job, msa_max_depth)

    summaries: list[dict] = []
    written: list[Path] = []
    for seed in seeds:
        print(f"Folding (seed={seed}, samples={n_samples})...")
        results = builder.fold(
            model,
            spec,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            num_diffusion_samples=n_samples,
            seed=seed,
            msa_max_depth=msa_max_depth,
            complex_id=job["job_name"],
        )
        # fold() returns a single result when num_diffusion_samples == 1.
        results = results if isinstance(results, list) else [results]
        for i, result in enumerate(results):
            # Tag files only when more than one structure is produced.
            tag = f"_seed{seed}_sample{i}" if multi else ""
            files, conf = write_outputs(result, job, tag=tag, seed=seed, sample=i)
            written.extend(files)
            summaries.append(conf)
            ptm = f"{conf['ptm']:.3f}" if conf["ptm"] is not None else "n/a"
            mp = f"{conf['mean_plddt']:.3f}" if conf["mean_plddt"] is not None else "n/a"
            print(f"  seed={seed} sample={i}: pTM={ptm}, mean pLDDT={mp}")

    # When multiple structures, write a ranking summary (best pTM first).
    if multi:
        ranked = sorted(
            summaries, key=lambda c: (c["ptm"] is not None, c["ptm"] or 0.0), reverse=True
        )
        summary_path = Path(job["output_path"]).expanduser() / f"{job['job_name']}_models.json"
        summary_path.write_text(json.dumps(ranked, indent=2))
        written.append(summary_path)
        best = ranked[0]
        print(
            f"\nDone. {len(summaries)} structures. Best: seed={best['seed']} "
            f"sample={best['sample']} pTM={best['ptm']}"
        )
    else:
        print("\nDone.")
    print(f"Wrote {len(written)} files to {job['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
