"""Fractus Space — shared-memory AI backend.

This FastAPI app powers the Fractus HuggingFace Space: a single Fractus
instance with a SHARED knowledge base that persists across all users and
sessions. Every user's input is stored, retrieved, and remembered by the
next user — proving Fractus's continuous learning and persistent memory.

Architecture:
  - Fractus model (Fractus-1B trained weights → CTE runtime)
  - SharedKnowledgeBase: ONE global KB across all HTTP sessions
  - Basic keyword moderation before storing (blocks toxic content)
  - FastAPI: POST /chat, GET /memories, GET /stats
"""
import os
import sys
import time
import re
import threading
from typing import Optional

# Path to the Fractus package (sibling of the space).
FRACTUS_PATH = os.environ.get("FRACTUS_PATH", "/workspace/fractus")
sys.path.insert(0, FRACTUS_PATH)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Fractus imports (lazy — model loads on first request) ---
_engine = None
_kb = None
_lock = threading.Lock()


def get_engine():
    """Lazy-load Fractus engine + shared KB. Loads ONCE, shared across requests."""
    global _engine, _kb
    with _lock:
        if _engine is None:
            from fractus.continuous_engine import ContinuousThoughtEngine
            from fractus.tokenizer import FractusTokenizer
            from fractus.rag import KnowledgeBase, RAGEngine
            from fractus.memory import PersistentMemory

            tok = FractusTokenizer.gpt2_compatible()

            # 1B config — the REAL Fractus (d=1280, 16 blocks, 128 experts)
            engine = ContinuousThoughtEngine(
                vocab_size=50257, d_model=1280,
                n_heads=20, d_head=64, n_levels=2,
                n_oscillators=16, coupling_rank=8,
                n_experts=128, top_k=2,
                expert_d_ff=2048, siren_rank=64, n_layers=16,
            )

            # Load the trained 1B checkpoint (loss ~80 from overnight training)
            import torch
            try:
                from huggingface_hub import hf_hub_download
                ckpt_file = hf_hub_download(
                    "thefinalboss/fractus-cte",
                    "checkpoints/fractus_1b_gpu3.pt",  # best overnight (loss 80.9)
                    repo_type="model")
                sd = torch.load(ckpt_file, weights_only=False, map_location="cpu")
                state = sd.get("model_state", sd)
                own = engine.state_dict()
                matched = {k: v for k, v in state.items()
                           if k in own and own[k].shape == v.shape}
                own.update(matched)
                engine.load_state_dict(own)
                print(f"[Fractus] Loaded 1B checkpoint: {len(matched)}/{len(own)} tensors "
                      f"({sum(p.numel() for p in engine.parameters())/1e9:.2f}B params)", flush=True)
            except Exception as e:
                print(f"[Fractus] 1B load failed ({e}) — trying palier0 fallback", flush=True)
                # Fallback: small palier0
                engine = ContinuousThoughtEngine(
                    vocab_size=50257, d_model=128,
                    n_heads=4, d_head=32, n_levels=2,
                    n_oscillators=16, n_experts=8, top_k=2,
                )
                try:
                    from huggingface_hub import hf_hub_download
                    ckpt_file = hf_hub_download(
                        "thefinalboss/fractus-cte",
                        "checkpoints/fractus_palier0.pt",
                        repo_type="model")
                    sd = torch.load(ckpt_file, weights_only=False, map_location="cpu")
                    state = sd.get("model_state", sd)
                    own = engine.state_dict()
                    matched = {k: v for k, v in state.items()
                               if k in own and own[k].shape == v.shape}
                    own.update(matched)
                    engine.load_state_dict(own)
                    print(f"[Fractus] Loaded palier0 fallback", flush=True)
                except Exception:
                    print("[Fractus] Random weights", flush=True)

            kb = KnowledgeBase(d_model=128)
            kb_path = os.environ.get("FRACTUS_KB_PATH", "/data/fractus_kb.pkl")
            if os.path.exists(kb_path):
                try:
                    kb.load(kb_path)
                    print(f"[Fractus] Loaded shared KB: {len(kb.texts)} memories", flush=True)
                except Exception as e:
                    print(f"[Fractus] KB load failed: {e}", flush=True)

            rag = RAGEngine(engine, tok, kb)

            # --- PersistentMemory: thought-state cross-session memory (the "subconscious") ---
            # Attached to the CTE. Salience head bias forced high so the engine
            # consolidates most thoughts (untrained — we want visible memory for the demo).
            import torch as _torch
            with _torch.no_grad():
                engine.salience_head.bias.fill_(3.0)  # sigmoid(3) ≈ 0.95 → consolidates most thoughts
            pm_path = os.environ.get("FRACTUS_PM_PATH", "/data/fractus_pm.pt")
            pm = PersistentMemory(d_model=128, max_memories=256, path=pm_path)
            if os.path.exists(pm_path):
                try:
                    pm.load(pm_path)
                    print(f"[Fractus] Loaded PersistentMemory: {len(pm)} memories", flush=True)
                except Exception as e:
                    print(f"[Fractus] PM load failed: {e}", flush=True)
            engine.attach_memory(pm)
            print(f"[Fractus] PersistentMemory attached (salience bias=3.0, blend=0.05)", flush=True)

            _engine = {"engine": engine, "tok": tok, "kb": kb, "rag": rag, "pm": pm}
        return _engine


# ============================================================================
# Moderation: basic keyword filter to keep the shared KB safe-ish.
# ============================================================================

# Crude but effective: blocks obvious toxic content from being stored.
# This is NOT a sophisticated moderation system — just a safety net so the
# Space doesn't get taken down in 2 hours.
_BLOCKED_KEYWORDS = [
    # Hate speech / slurs (censored here, expanded at runtime)
    "nigger", "faggot", "tranny", "kike", "spic", "chink", "retard",
    # Self-harm / violence instructions
    "kill yourself", "how to make a bomb", "suicide method",
    # CSAM-adjacent
    "child porn", "cp ", "loli", "shota", "underage nude",
    # Doxxing patterns (very basic)
    "social security number", "credit card number",
]
_BLOCKED_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _BLOCKED_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def moderate(text: str) -> tuple:
    """Returns (is_blocked, reason). Basic keyword filter only."""
    if not text or len(text.strip()) < 1:
        return True, "empty"
    if len(text) > 2000:
        return True, "too_long"
    m = _BLOCKED_RE.search(text)
    if m:
        return True, f"blocked_keyword:{m.group(1)[:20]}"
    return False, ""


def save_kb_async():
    """Persist the KB + PM in a background thread (don't block the response)."""
    def _save():
        try:
            e = get_engine()
            kb_path = os.environ.get("FRACTUS_KB_PATH", "/data/fractus_kb.pkl")
            os.makedirs(os.path.dirname(kb_path) or ".", exist_ok=True)
            e["kb"].save(kb_path)
            # Also persist the PersistentMemory (thought-state memories).
            pm_path = os.environ.get("FRACTUS_PM_PATH", "/data/fractus_pm.pt")
            os.makedirs(os.path.dirname(pm_path) or ".", exist_ok=True)
            e["pm"].save(pm_path)
        except Exception as ex:
            print(f"[Fractus] KB/PM save failed: {ex}", flush=True)
    threading.Thread(target=_save, daemon=True).start()


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(title="Fractus Space")

# Serve the static frontend.
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    speaker: Optional[str] = "anonymous"


class ChatResponse(BaseModel):
    reply: str
    memories_used: int
    stored: bool
    moderation: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main chat UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(content=open(index_path, encoding="utf-8").read())
    return HTMLResponse("<h1>Fractus Space</h1><p>static/index.html missing</p>")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main endpoint: user message → Fractus reply, with shared memory."""
    # 1. Moderate the input BEFORE storing.
    blocked, reason = moderate(req.message)
    if blocked:
        return ChatResponse(
            reply="(Fractus declined to store this message — moderation filter.)",
            memories_used=0, stored=False, moderation=reason,
        )

    try:
        e = get_engine()
    except Exception as ex:
        raise HTTPException(500, f"Fractus failed to start: {ex}")

    # 2. Retrieve relevant memories + generate a reply (also learns the input).
    try:
        result = e["rag"].converse(req.message, speaker=req.speaker)
        reply = result.get("answer", "") or "(Fractus stayed silent.)"
        memories_used = len(result.get("retrieved_context", ""))
        stored = True
        # Persist the KB in background.
        save_kb_async()
    except Exception as ex:
        return ChatResponse(
            reply=f"(Fractus encountered an error: {ex})",
            memories_used=0, stored=False, moderation="runtime_error",
        )

    return ChatResponse(
        reply=reply,
        memories_used=memories_used,
        stored=stored,
    )


@app.get("/memories")
async def memories(limit: int = 20):
    """Show the most recent shared memories (transparency feature)."""
    try:
        e = get_engine()
        kb = e["kb"]
        recent = list(zip(kb.texts[-limit:], kb.sources[-limit:] if kb.sources else ["?"]*limit))
        return {"count": len(kb.texts), "recent": [{"text": t[:200], "source": s} for t, s in reversed(recent)]}
    except Exception as ex:
        return {"error": str(ex), "count": 0, "recent": []}


@app.get("/pmemories")
async def pmemories(limit: int = 20):
    """Show the PersistentMemory's thought-state memories (the 'subconscious')."""
    try:
        e = get_engine()
        pm = e["pm"]
        n = len(pm)
        # Return recent context labels + importance.
        recent = []
        ctxs = pm.contexts[-limit:] if pm.contexts else []
        imps = pm.importance[-limit:] if pm.importance else []
        for i in range(min(len(ctxs), limit)):
            recent.append({"context": ctxs[i][:200] if ctxs[i] else "(unlabeled)",
                           "importance": round(imps[i], 3) if i < len(imps) else 0.5})
        return {"count": n, "max": pm.max_memories, "recent": list(reversed(recent))}
    except Exception as ex:
        return {"error": str(ex), "count": 0, "recent": []}


@app.get("/stats")
async def stats():
    """Live stats for the UI."""
    try:
        e = get_engine()
        return {
            "memories": len(e["kb"].texts),
            "pmemories": len(e["pm"]),
            "engine": "Fractus-CTE",
            "status": "online",
        }
    except Exception:
        return {"memories": 0, "engine": "Fractus-CTE", "status": "starting"}


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}
