#!/bin/bash -l
#SBATCH --job-name=memviz_big
#SBATCH --partition=A100
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --exclude=gpu07
#SBATCH --output=/home/mjkang/optimization/esm/bench_results/ncu/memviz_big_%j.log
cd /home/mjkang/optimization/esm/.claude/worktrees/memtrace
export CUDA_HOME=/usr/local/cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/mjkang/optimization/esm/.venv/bin/python
SNAP=/home/mjkang/optimization/esm/bench_results/diffab/snapshots
set -x
# Re-record 1024/2048 with offload-LM (matching the 512 memviz runs) and a
# ring big enough that it does NOT wrap (400k wrapped for these lengths).
for cfg in fp32 bf16-triton bf16-cuda; do
  $PY scripts/memviz_capture.py --input /home/mjkang/optimization/esm/bench_results/H2343_local.json \
    --bucket 1024 --attn $cfg --offload-lm --max-entries 4000000 \
    --out $SNAP/memviz_1024_${cfg}_offload.pickle 2>&1 | grep -viE "libcue|libcuda"
  echo "RUN 1024 $cfg exit: ${PIPESTATUS[0]}"
done
for cfg in fp32 bf16-triton bf16-cuda; do
  $PY scripts/memviz_capture.py --input /home/mjkang/optimization/esm/bench_results/H1382_local.json \
    --bucket 2048 --attn $cfg --offload-lm --max-entries 10000000 \
    --out $SNAP/memviz_2048_${cfg}_offload.pickle 2>&1 | grep -viE "libcue|libcuda"
  echo "RUN 2048 $cfg exit: ${PIPESTATUS[0]}"
done
echo "MEMVIZ BIG DONE"
