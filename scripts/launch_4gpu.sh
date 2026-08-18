#!/bin/bash
# Launch Fractus 1B training on 4 GPUs
set -e
cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

for i in 0 1 2 3; do
  echo "Starting GPU $i..."
  GPU_ID=$i CUDA_VISIBLE_DEVICES=$i setsid python -u scripts/train_1b_multi_gpu.py > logs/gpu$i.log 2>&1 < /dev/null &
done

echo "All 4 GPUs launched. Monitor with: tail -f logs/gpu*.log"
