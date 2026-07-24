#!/usr/bin/env bash
# Profile the ESMFold2 trunks (pair trunk + diffusion trunk) with Nsight Systems.
#
# This wraps `scripts/nsys_profile_esmfold2.py` in `nsys profile`. The Python
# side lays down NVTX ranges (pair_trunk, PairUpdateBlock[i], msa_encoder,
# MSAEncoderBlock[i], diffusion_step, esm-c) and brackets the measured fold in
# cudaProfilerStart/Stop; nsys collects everything else:
#
#   requirement                         provided by
#   ----------------------------------  --------------------------------------
#   Utilization per unit                NVTX ranges  -> per-region SM/kernel time
#                                       (nvtx_pushpop_sum + gpu-metrics rows)
#   GPU utilization                     --gpu-metrics-devices (SM %, DRAM %,
#                                       tensor-core %, sampled on-device)
#   CPU + GPU timeline, calls (h2g)     --trace=cuda,nvtx,osrt,cublas,cudnn
#                                       (osrt = CPU/OS runtime timeline; cuda =
#                                       GPU kernels + HtoD/"h2g" memcpys + API
#                                       call rows correlated CPU->GPU)
#   Memory operation time (GPU)         --cuda-memory-usage=true + the
#                                       cuda_gpu_mem_time_sum / _size_sum reports
#
# Usage:
#   srun --gres=gpu:1 --pty scripts/nsys_profile_esmfold2.sh examples/H2343.json
#   srun --gres=gpu:1 --pty scripts/nsys_profile_esmfold2.sh examples/H2343.json \
#        float32 4 20             # <input> <dtype> <num_loops> <num_sampling_steps>
#
# Positional args (all optional except the input JSON):
#   $1  input job JSON            (default: examples/job.json)
#   $2  dtype                     (default: float32 — the only supported value;
#                                  ESMFold2 does bf16 internally via autocast)
#   $3  num_loops (pair-trunk recycles)      (default: 4  — keep small for nsys)
#   $4  num_sampling_steps (diffusion steps) (default: 20 — keep small for nsys)
#   $5  output .nsys-rep basename (default: nsys_esmfold2_<jobstem>)
#
# Keep the loop / step counts SMALL: a full run (16 loops x 200 steps) makes a
# multi-GB report. A few of each is enough to profile per-unit behaviour, and
# the trunks repeat identical work per iteration/step anyway.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INPUT="${1:-examples/job.json}"
DTYPE="${2:-float32}"
NUM_LOOPS="${3:-4}"
NUM_STEPS="${4:-20}"
JOBSTEM="$(basename "${INPUT%.*}")"
OUT="${5:-nsys_esmfold2_${JOBSTEM}}"

# Locate nsys (module env may not put it on PATH).
NSYS="$(command -v nsys || true)"
for cand in \
  /opt/nvidia/nsight-systems/*/bin/nsys \
  /usr/local/cuda/bin/nsys; do
  [ -z "$NSYS" ] && [ -x "$cand" ] && NSYS="$cand"
done
[ -z "$NSYS" ] && { echo "ERROR: nsys not found on PATH or in /opt/nvidia/nsight-systems." >&2; exit 1; }

# Use the project venv's python if present, else whatever python3 is active.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "nsys:   $NSYS ($("$NSYS" --version | head -1))"
echo "python: $PY"
echo "input:  $INPUT  dtype=$DTYPE  loops=$NUM_LOOPS  steps=$NUM_STEPS"
echo "output: ${OUT}.nsys-rep"
echo

# Hardware GPU/SM-utilization sampling (--gpu-metrics-devices) needs elevated
# perf-counter permission. On this cluster it fails with ERR_NVGPUCTRPERM
# (NVreg_RestrictProfilingToAdminUsers=1), so it is OFF by default. Enable it
# once an admin lifts that restriction:  GPU_METRICS=1 srun ... nsys_profile...sh
# (optionally GPU_METRICS_DEVICES=all|<id> to pick the device set).
GPU_METRICS_ARGS=()
if [ "${GPU_METRICS:-0}" = "1" ]; then
  GPU_METRICS_ARGS=(--gpu-metrics-devices="${GPU_METRICS_DEVICES:-cuda-visible}")
  echo "GPU metrics sampling: ON (${GPU_METRICS_DEVICES:-cuda-visible})"
else
  echo "GPU metrics sampling: OFF (set GPU_METRICS=1 to enable; needs admin perf-counter access)."
fi

# --capture-range=cudaProfilerApi  -> record only between cudaProfilerStart/Stop
#     (the measured fold), skipping model load + the warm-up pass.
# --cuda-memory-usage=true          -> per-op GPU memory-operation timing/size.
# --trace=cuda,nvtx,osrt,cublas,cudnn -> GPU kernels + HtoD/DtoH copies + CPU
#     (osrt) timeline + library calls, with NVTX region overlay + CPU->GPU
#     correlation ("call h2g").
"$NSYS" profile \
  --output="$OUT" \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-memory-usage=true \
  ${GPU_METRICS_ARGS[@]+"${GPU_METRICS_ARGS[@]}"} \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cudabacktrace=none \
  --python-sampling=true \
  "$PY" scripts/nsys_profile_esmfold2.py \
    --input "$INPUT" \
    --dtype "$DTYPE" \
    --num-loops "$NUM_LOOPS" \
    --num-sampling-steps "$NUM_STEPS" \
    --warmup 1 \
    --offload-lm

echo
echo "=== nsys summary (per-unit / GPU-mem / kernel time) ==="
# nvtx_pushpop_sum   : time per NVTX region (per unit: pair_trunk, blocks, diffusion_step)
# cuda_gpu_kern_sum  : GPU kernel time  (GPU utilization by kernel)
# cuda_gpu_mem_time_sum / _size_sum : GPU memory-operation time + bytes (incl. HtoD "h2g")
"$NSYS" stats \
  --report nvtx_pushpop_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  "${OUT}.nsys-rep" || true

echo
echo "Open the full CPU/GPU timeline + GPU-utilization rows in the Nsight Systems GUI:"
echo "   nsys-ui ${OUT}.nsys-rep"
