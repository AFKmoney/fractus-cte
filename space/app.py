"""Fractus Space — ZeroGPU (A100) powered 1B Continuous Cognitive Agent."""
import os
import sys
import torch

# Install fractus from the model repo if not present
try:
    from fractus.continuous_engine import ContinuousThoughtEngine
except ImportError:
    from huggingface_hub import snapshot_download
    snapshot_download("thefinalboss/fractus-cte", repo_type="model",
                      allow_patterns=["fractus/**"], local_dir="/tmp/fractus_src")
    sys.path.insert(0, "/tmp/fractus_src")
    from fractus.continuous_engine import ContinuousThoughtEngine

import gradio as gr
import spaces  # ZeroGPU — gives A100 access on demand

CHECKPOINT = "checkpoints/fractus_1b_gpu3.pt"
TARGET_1B = dict(
    vocab_size=50257, d_model=1280, n_heads=20, d_head=64, n_levels=2,
    n_oscillators=16, coupling_rank=8, n_experts=128, top_k=2,
    expert_d_ff=2048, siren_rank=64, n_layers=16,
)

_engine = None
_tok = None


def load_model():
    """Load the 1B on CPU once, moved to GPU per-request by @spaces.GPU."""
    global _engine, _tok
    if _engine is not None:
        return _engine, _tok

    from fractus.tokenizer import FractusTokenizer
    _tok = FractusTokenizer.gpt2_compatible()

    from huggingface_hub import hf_hub_download
    from fractus.grow import grow_cte
    ckpt_path = hf_hub_download("thefinalboss/fractus-cte", CHECKPOINT, repo_type="model")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})

    # Try direct load; if config matches, use it, else grow from seed
    try:
        seed = ContinuousThoughtEngine(
            vocab_size=cfg.get("vocab_size", 50257),
            d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 2),
            d_head=cfg.get("d_head", 64), n_layers=cfg.get("n_layers", 2),
            n_levels=2, n_oscillators=cfg.get("n_oscillators", 8),
            coupling_rank=cfg.get("coupling_rank", 4),
            n_experts=cfg.get("n_experts", 4), top_k=cfg.get("top_k", 2),
            expert_d_ff=cfg.get("expert_d_ff", 128),
            siren_rank=cfg.get("siren_rank", 32),
        )
        seed.load_state_dict(ckpt["model_state"])
        engine = grow_cte(seed, TARGET_1B)  # grow to 1B
    except Exception:
        # Build fresh and load shape-matched
        engine = ContinuousThoughtEngine(**TARGET_1B)
        own = engine.state_dict()
        for k, v in ckpt["model_state"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        engine.load_state_dict(own)

    engine.eval()
    _engine = engine
    print(f"[Fractus] 1B loaded: {sum(p.numel() for p in engine.parameters()):,} params")
    return _engine, _tok


@spaces.GPU  # ← ZeroGPU: A100 allocated for this function's duration
def generate(prompt: str, max_tokens: int = 80, temperature: float = 0.8) -> str:
    """Generate with the 1B on A100."""
    engine, tok = load_model()
    device = torch.device("cuda")
    engine = engine.to(device)
    engine.reset_thought(batch_size=1)

    # Feed the prompt
    ids = tok.encode(prompt)[:32]
    for t in ids:
        engine.tick(torch.tensor([t], device=device))

    # Generate
    generated = list(ids)
    cur = ids[-1] if ids else 0
    for _ in range(max_tokens):
        with torch.no_grad():
            logits, conf = engine.tick(torch.tensor([cur], device=device))
        # Sample from top-k
        l = logits[0] / max(temperature, 1e-8)
        topv, topi = l.topk(40)
        probs = torch.softmax(topv, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        nxt = topi[idx].item()
        if nxt == 50256:
            break
        generated.append(nxt)
        cur = nxt

    engine = engine.to("cpu")  # free the GPU
    return tok.decode(generated)


def chat(message: str, history: list) -> str:
    """Gradio chat handler."""
    if not message.strip():
        return ""
    try:
        return generate(message)
    except Exception as e:
        return f"(error: {e})"


# --- UI ---
demo = gr.ChatInterface(
    fn=chat,
    title="🧠 Fractus",
    description="""**Continuous Cognitive Agent** — not a transformer.
A dynamical system that thinks tick by tick, routed by Kuramoto oscillators.
1B params, trained on 4.23B tokens (neuroscience, esoteric, code, identity).""",
    examples=["Hello, who are you?", "What are you?", "Tell me about yourself"],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
