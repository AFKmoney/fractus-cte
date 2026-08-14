"""Fractus Space — ZeroGPU (A100) powered 1B Continuous Cognitive Agent."""
import os
import sys
import torch

# --- fractus package: try local, then download from HF ---
try:
    from fractus.continuous_engine import ContinuousThoughtEngine
except ImportError:
    from huggingface_hub import snapshot_download
    snapshot_download("thefinalboss/Fractus-Space", repo_type="space",
                      allow_patterns=["fractus/**"], local_dir="/tmp/fsrc")
    sys.path.insert(0, "/tmp/fsrc")
    from fractus.continuous_engine import ContinuousThoughtEngine

from fractus.tokenizer import FractusTokenizer
import gradio as gr
import spaces  # ZeroGPU

TARGET_1B = dict(
    vocab_size=50257, d_model=1280, n_heads=20, d_head=64, n_levels=2,
    n_oscillators=16, coupling_rank=8, n_experts=128, top_k=2,
    expert_d_ff=2048, siren_rank=64, n_layers=16,
)

# --- Load model on CPU at STARTUP (not inside GPU function) ---
print("[Fractus] Loading tokenizer...", flush=True)
tok = FractusTokenizer.gpt2_compatible()

print("[Fractus] Loading 1B checkpoint on CPU...", flush=True)
engine = None
try:
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        "thefinalboss/fractus-cte",
        "checkpoints/fractus_1b_gpu3.pt",
        repo_type="model")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # Build the 1B directly and shape-match load
    engine = ContinuousThoughtEngine(**TARGET_1B)
    own = engine.state_dict()
    matched = 0
    for k, v in ckpt.get("model_state", ckpt).items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            matched += 1
    engine.load_state_dict(own)
    engine.eval()
    print(f"[Fractus] 1B loaded: {matched}/{len(own)} tensors, "
          f"{sum(p.numel() for p in engine.parameters())/1e9:.2f}B params", flush=True)
except Exception as e:
    print(f"[Fractus] 1B load failed: {e}", flush=True)
    # Fallback: small model so the Space still works
    engine = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=4, d_head=32, n_levels=2,
        n_oscillators=16, n_experts=8, top_k=2, expert_d_ff=128, siren_rank=32, n_layers=2)
    try:
        from huggingface_hub import hf_hub_download as dl
        small = dl("thefinalboss/fractus-cte", "checkpoints/fractus_palier0.pt", repo_type="model")
        sd = torch.load(small, weights_only=False, map_location="cpu")
        own = engine.state_dict()
        for k, v in sd.get("model_state", sd).items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
        engine.load_state_dict(own)
    except Exception:
        pass
    engine.eval()
    print("[Fractus] Fallback small model loaded", flush=True)


@spaces.GPU(duration=120)
def generate(prompt: str, max_tokens: int = 60, temperature: float = 0.8) -> str:
    """Generation on ZeroGPU (A100). If GPU unavailable, falls back to CPU."""
    global engine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = engine.to(device)
    engine.reset_thought(batch_size=1)

    try:
        ids = tok.encode(prompt)[:32]
        for t in ids:
            engine.tick(torch.tensor([t], device=device))

        generated = list(ids)
        cur = ids[-1] if ids else 0
        for _ in range(max_tokens):
            with torch.no_grad():
                logits, conf = engine.tick(torch.tensor([cur], device=device))
            l = logits[0] / max(temperature, 1e-8)
            topv, topi = l.topk(40)
            probs = torch.softmax(topv, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            nxt = topi[idx].item()
            if nxt == 50256:
                break
            generated.append(nxt)
            cur = nxt
        return tok.decode(generated)
    finally:
        if device.type == "cuda":
            engine = engine.to("cpu")  # free the GPU


def chat(message: str, history: list) -> str:
    if not message.strip():
        return ""
    try:
        return generate(message)
    except Exception as e:
        return f"(error: {str(e)[:150]})"


demo = gr.ChatInterface(
    fn=chat,
    title="🧠 Fractus",
    description="""**Continuous Cognitive Agent** — not a transformer.
Thinks tick by tick, routed by Kuramoto oscillators. 1B params on 4.23B tokens.""",
    examples=["Hello, who are you?", "What are you?", "Tell me something"],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
