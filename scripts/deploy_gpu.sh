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
git clone https://github.com/AFKmoney/fractus-cte.git
cd fractus-cte

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

# 5. Build combined corpus from all datasets
echo ""
echo "=== Building training corpus ==="
python -c "
import torch, os, glob, json
from fractus.tokenizer import FractusTokenizer
tok = FractusTokenizer.gpt2_compatible()

# Collect all .pt datasets.
pt_files = sorted(glob.glob('data/hf_datasets/datasets/*.pt'))
print(f'Found {len(pt_files)} pre-tokenized datasets')

all_tokens = []
total = 0
for f in pt_files:
    try:
        t = torch.load(f, weights_only=False)
        all_tokens.append(t)
        total += len(t)
        print(f'  {os.path.basename(f)}: {len(t):,} tokens')
    except Exception as e:
        print(f'  SKIP {f}: {e}')

# Also tokenize raw JSONL datasets.
for jsonl_dir in ['data/hf_datasets/neuro_code_math', 'data/hf_datasets/cognitive_skills']:
    if os.path.exists(jsonl_dir):
        files = sorted(glob.glob(os.path.join(jsonl_dir, '*.jsonl')))
        for jf in files[:5]:  # first 5 files per dir for speed
            texts = []
            with open(jf, 'r', encoding='utf-8', errors='ignore') as fh:
                for line in fh:
                    try:
                        entry = json.loads(line.strip())
                        if 'messages' in entry:
                            text = ' '.join(m.get('content','') for m in entry['messages'])
                        elif 'instruction' in entry:
                            text = entry.get('instruction','') + ' ' + entry.get('output','')
                        elif 'text' in entry:
                            text = entry['text']
                        else:
                            continue
                        if text and len(text) > 20:
                            texts.append(text)
                    except:
                        pass
            if texts:
                combined = '\\n\\n'.join(texts[:200])
                tokens = tok.encode(combined)
                all_tokens.append(torch.tensor(tokens, dtype=torch.int32))
                total += len(tokens)

print(f'\\nTotal tokens: {total:,}')
mega = torch.cat(all_tokens)
g = torch.Generator().manual_seed(42)
# Cap at 1B tokens: uses all available data (the HF repo holds ~1B+),
# and a 1B-element permutation fits in ~8GB RAM on a GPU box.
# The 1B training stage requests 2B tokens — it will iterate over
# everything available here.
perm = torch.randperm(min(len(mega), 1_000_000_000), generator=g)
mega = mega[perm].to(torch.int32)
torch.save(mega, 'data/training_corpus.pt')
print(f'Saved training_corpus.pt: {len(mega):,} tokens ({os.path.getsize(\"data/training_corpus.pt\")/1e6:.0f}MB)')
"

# 6. Train paliers 0-3 on CPU (if no checkpoint exists)
if [ ! -f "checkpoints/fractus_palier3.pt" ]; then
    echo ""
    echo "=== Training paliers 0-3 (CPU) ==="
    # Update train_progressive to use the new corpus.
    export FRACTUS_CORPUS="data/training_corpus.pt"
    python scripts/train_progressive.py --paliers 0,1,2,3 --accumulation-steps 8
else
    echo ""
    echo "=== Checkpoint exists, skipping paliers 0-3 ==="
fi

# 7. Grow to 1B + train on GPU
echo ""
echo "=== GPU Training: 1B ==="
echo "Loading palier 3 checkpoint, growing to 1B..."
python scripts/train_1b_gpu.py \
    --checkpoint checkpoints/fractus_palier3.pt \
    --tokens 2000000000 \
    --batch-size 8 \
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
