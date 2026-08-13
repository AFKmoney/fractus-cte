#!/usr/bin/env bash
# Fractus 1B GPU Deployment Script
# Run this on the GPU machine after SSH'ing in.
#
# Usage:
#   bash deploy_gpu.sh
#
# What it does:
#   1. Installs dependencies
#   2. Clones fractus-cte + downloads datasets from HF
#   3. Trains paliers 0-3 on CPU (if no checkpoint)
#   4. Grows to 1B + trains on GPU
#
set -e

echo "============================================"
echo "  FRACTUS 1B GPU DEPLOYMENT"
echo "============================================"

# 1. Check GPU
echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 1b. Preflight: disk + HF token (catch problems before the long download/train).
echo "=== Preflight ==="
FREE_GB=$(df -BG . 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G')
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -lt 20 ]; then
    echo "WARNING: only ${FREE_GB}GB free on disk — need ~20GB (3GB datasets + 4GB corpus + ~8GB checkpoints)."
    echo "         Continuing, but the run may fail mid-training."
else
    echo "  disk: ${FREE_GB:-?}GB free (need ~20GB) — OK"
fi
if [ -z "$HF_TOKEN" ]; then
    echo "  HF_TOKEN: NOT SET — the 1B checkpoint upload to HF will be skipped."
    echo "           Set it with: export HF_TOKEN=hf_xxx (from https://huggingface.co/settings/tokens)"
else
    echo "  HF_TOKEN: set — 1B will upload to thefinalboss/Fractus-1B"
fi
python -c "import sys; print(f'  python: {sys.version.split()[0]}')" || echo "  python: MISSING"
echo ""

# 2. Install dependencies
echo "=== Installing dependencies ==="
pip install torch numpy tokenizers matplotlib huggingface_hub --quiet
echo "Done"

# 3. Clone fractus-cte
echo ""
echo "=== Cloning fractus-cte ==="
git clone https://github.com/AFKmoney/fractus-cte.git 2>/dev/null || true
cd fractus-cte || { echo "fractus-cte dir missing, aborting"; exit 1; }

# 4. Download datasets from HF
echo ""
echo "=== Downloading datasets from HF ==="
python -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('data', exist_ok=True)
snapshot_download(
    repo_id='thefinalboss/fractus-datasets',
    repo_type='dataset',
    local_dir='data/hf_datasets')
print('Datasets downloaded')
"

# 5. Build combined corpus from ALL datasets (streaming tokenize of every .jsonl)
echo ""
echo "=== Building training corpus (3B cap = sparse-MoE Chinchilla for ~119M active params) ==="
python scripts/build_corpus.py --src data/hf_datasets --out data/training_corpus.pt --cap 3000000000

# 6. Train paliers 0-3 (GPU if available, else CPU — auto-detected)
if [ ! -f "checkpoints/fractus_palier3.pt" ]; then
    echo ""
    echo "=== Training paliers 0-3 (auto: GPU if present) ==="
    export FRACTUS_CORPUS="data/training_corpus.pt"
    python scripts/train_progressive.py --paliers 0,1,2,3 --accumulation-steps 8
else
    echo ""
    echo "=== Checkpoint exists, skipping paliers 0-3 ==="
fi

# 7. Grow to 1B + train on GPU (A100 80GB crunch config)
#    NOTE: --batch-size is currently vestigial (loop runs B=1 chunk); the real
#    throughput lever is --seq-len (more tokens per forward = better GPU util).
#    3B tokens = sparse-MoE Chinchilla for ~119M active params.
echo ""
echo "=== GPU Training: 1B (A100 80GB crunch config) ==="
echo "Loading palier 3 checkpoint, growing to 1B..."
python scripts/train_1b_gpu.py \
    --checkpoint checkpoints/fractus_palier3.pt \
    --tokens 3000000000 \
    --seq-len 128 \
    --bf16 \
    --accumulation-steps 4 \
    --corpus data/training_corpus.pt

echo ""
echo "============================================"
echo "  FRACTUS 1B TRAINING COMPLETE"
echo "============================================"
echo ""
echo "Checkpoint: checkpoints/fractus_1b_gpu.pt"
echo "Run 'python -c \"from fractus.continuous_engine import ContinuousThoughtEngine; e = ContinuousThoughtEngine.from_pretrained(\\\"checkpoints/fractus_1b_gpu.pt\\\")\"' to load."
