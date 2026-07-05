#!/usr/bin/env bash
# Set up the Python environment for the modified ESMFold2 runner
# (scripts/run_esmfold2.py, scripts/profile_esmfold2.py).
#
# Reproduces the known-good stack:
#   - Python 3.12 venv, uv-managed, at $REPO_ROOT/.venv
#   - esm (this repo, editable) + its deps, incl. the Biohub `transformers`
#     fork that ships `transformers.models.esmfold2` (pinned in pyproject.toml)
#   - torch 2.8.0+cu126   (NOT the latest: torch.compile for ESMFold2 breaks on
#                          2.12; Biohub CI pins 2.8.0/cu126)
#   - flash-attn          (HARD dep — esmc imports it unguarded; must be built
#                          against the installed torch, so it comes AFTER torch)
#   - cuequivariance-torch + cuequivariance-ops-torch-cu12  (trimul kernels,
#                          ~7.7x; the -ops wheel lives on the NVIDIA index)
#   - rdkit               (ligand SMILES/CCD conformers; pulled by pyproject)
#   - matplotlib          (the PAE plot in run_esmfold2.py)
#
# Requirements:
#   - `uv` on PATH (https://docs.astral.sh/uv/)
#   - CUDA 12.x driver; a machine with `nvcc` for the flash-attn source build
#   - Run the VERIFY step (and any fold) on a GPU node: cuequivariance /
#     flash-attn need libcuda.so.1 / libnvrtc, absent on login nodes.
#
# Usage:
#   bash scripts/setup_env.sh              # full setup
#   bash scripts/setup_env.sh --verify     # only re-run the import checks
#
# NOTE on the uv + anaconda gotcha (this cluster): the login shell auto-activates
# a shared anaconda; a bare `uv pip install` targets THAT env and fails. Every
# install below is pinned with `--python "$PY"` and VIRTUAL_ENV, so it always
# lands in the project venv.

set -euo pipefail

# --------------------------- config (edit if needed) ------------------------ #
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"

PYTHON_VERSION="3.12"
TORCH_VERSION="2.8.0+cu126"
CUDA_TAG="cu126"
FLASH_ATTN_VERSION="2.8.3.post1"
CUEQ_VERSION="0.10.0"

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
NVIDIA_INDEX="https://pypi.nvidia.com"

export VIRTUAL_ENV="$VENV"   # so uv resolves against the project venv, not anaconda

# uv pip install, always pinned to the project interpreter.
pip_install() { uv pip install --python "$PY" "$@"; }

# ------------------------------- verify step -------------------------------- #
verify() {
  echo "=== verifying environment ($PY) ==="
  "$PY" - <<'PYEOF'
import importlib, importlib.metadata as md
ok = True
def show(pkg, dist=None):
    global ok
    try:
        v = md.version(dist or pkg)
        print(f"  {pkg:34s} {v}")
    except Exception as e:
        print(f"  {pkg:34s} MISSING ({e})"); ok = False

import sys
print("python", sys.version.split()[0])
for p in ("torch","transformers","flash-attn","cuequivariance-torch",
          "cuequivariance-ops-torch-cu12","rdkit","matplotlib","numpy",
          "huggingface-hub","accelerate"):
    show(p)

# torch CUDA build + import-level checks (the ones that actually gate a fold)
import torch
print("  torch cuda build:", torch.version.cuda, "| available:", torch.cuda.is_available())
try:
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa
    from transformers.models.esmfold2.modeling_esmfold2_common import CUE_AVAILABLE
    print("  transformers.models.esmfold2: OK | CUE_AVAILABLE =", CUE_AVAILABLE)
except Exception as e:
    print("  transformers.models.esmfold2: FAIL:", e); ok = False
try:
    from esm.models.esmfold2 import ESMFold2InputBuilder, LigandInput  # noqa
    print("  esm.models.esmfold2 (LigandInput): OK")
except Exception as e:
    print("  esm.models.esmfold2: FAIL:", e); ok = False

# GPU-only deps: import checks (will fail on a login node — that's expected there)
for mod in ("flash_attn", "cuequivariance_ops_torch"):
    try:
        importlib.import_module(mod); print(f"  import {mod}: OK")
    except Exception as e:
        print(f"  import {mod}: FAIL (expected on login nodes): {e}")

print("=== verify:", "OK" if ok else "PROBLEMS FOUND", "===")
sys.exit(0 if ok else 1)
PYEOF
}

if [[ "${1:-}" == "--verify" ]]; then
  verify; exit $?
fi

# ------------------------------- 0. checks ---------------------------------- #
command -v uv >/dev/null 2>&1 || { echo "ERROR: 'uv' not found on PATH."; exit 1; }
echo "Repo:   $REPO_ROOT"
echo "Venv:   $VENV"

# ------------------------------- 1. venv ------------------------------------ #
if [[ ! -x "$PY" ]]; then
  echo "=== [1/6] creating venv (python $PYTHON_VERSION) ==="
  uv venv --python "$PYTHON_VERSION" "$VENV"
else
  echo "=== [1/6] venv exists, reusing $VENV ==="
fi

# --------------------- 2. project (+ transformers fork) --------------------- #
# Pulls the Biohub transformers fork, rdkit, biotite, accelerate, etc. This may
# pull a default torch; step 3 pins the correct CUDA build over it.
echo "=== [2/6] installing esm (editable) + deps ==="
pip_install -e "$REPO_ROOT"

# ------------------------------ 3. torch pin -------------------------------- #
# Must precede flash-attn so that extension builds against the final torch.
echo "=== [3/6] pinning torch $TORCH_VERSION ==="
pip_install "torch==${TORCH_VERSION}" \
  --index-url "$TORCH_INDEX" \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match

# ------------------------------ 4. flash-attn ------------------------------- #
# Hard dependency of esmc (imported unguarded). Rebuild against torch 2.8; ABI
# breaks across torch versions. Needs build helpers + nvcc for a source build.
echo "=== [4/6] installing flash-attn $FLASH_ATTN_VERSION (builds against torch) ==="
pip_install ninja packaging wheel setuptools psutil
pip_install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

# --------------------------- 5. cuequivariance ------------------------------ #
# Trimul + attention_pair_bias kernels. The -ops wheel is only on the NVIDIA
# index. run_esmfold2.py:_enable_cueq_ops() preloads libcue_ops.so at runtime,
# so no LD_LIBRARY_PATH is needed.
echo "=== [5/6] installing cuequivariance $CUEQ_VERSION ==="
pip_install \
  "cuequivariance-torch==${CUEQ_VERSION}" \
  "cuequivariance-ops-torch-cu12==${CUEQ_VERSION}" \
  --extra-index-url "$NVIDIA_INDEX" \
  --index-strategy unsafe-best-match

# ------------------------------ 6. matplotlib ------------------------------- #
# Required by run_esmfold2.py's PAE plot (matplotlib is only in pyproject's dev
# feature, so install it explicitly for the runtime env).
echo "=== [6/6] installing matplotlib (PAE plot) ==="
pip_install matplotlib

# --------------------------------- verify ----------------------------------- #
verify || echo "NOTE: GPU-only import failures above are expected on a login node."

cat <<EOF

Done. Environment ready at: $VENV

Weights & CCD: the biohub/ESMFold2 checkpoint (~24 GB) and ccd.pkl download to
  \$HF_HOME (default ~/.cache/huggingface) on first fold. To point CCD elsewhere:
  export ESMCFOLD_CCD_PATH=/path/to/ccd.pkl

Run a fold on a GPU node (float32 is the reference path; bf16 is broken), e.g.:
  srun -p gpu --gres=gpu:1 --mem=128G \\
    $PY scripts/run_esmfold2.py --input job.json --dtype float32
EOF
