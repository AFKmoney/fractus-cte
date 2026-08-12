"""Fractus Sandbox — Real-time control cockpit for the CTE.

Every metric exposed. Every control available. No black boxes.

Run: uvicorn app:app --host 0.0.0.0 --port 7860
"""
import os, sys, json, asyncio, threading, time, math
from typing import Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.memory import PersistentMemory
from fractus.cognitive_modes import CognitiveModes
from fractus.tokenizer import FractusTokenizer

app = FastAPI(title="Fractus Sandbox")

HERE = Path(__file__).parent
STATIC = HERE / "static"
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ─── Global state ────────────────────────────────────────────────────────────

_lock = threading.Lock()
_engine: Optional[ContinuousThoughtEngine] = None
_tokenizer: Optional[FractusTokenizer] = None
_memory: Optional[PersistentMemory] = None
_modes: Optional[CognitiveModes] = None
_trainer = None
_losses: list = []
_tick_log: list = []  # recent tick history for live view
_ws_clients: set = set()


def get_engine():
    global _engine, _tokenizer, _memory, _modes, _trainer
    if _engine is None:
        with _lock:
            if _engine is None:
                _tokenizer = FractusTokenizer.gpt2_compatible()
                _engine = ContinuousThoughtEngine(
                    vocab_size=50257, d_model=128, n_heads=4, d_head=32,
                    n_layers=2, n_levels=2, n_oscillators=8, coupling_rank=4,
                    n_experts=8, top_k=2, expert_d_ff=128, siren_rank=32,
                )
                _engine.reset_thought(batch_size=1)
                _memory = PersistentMemory(d_model=128, max_memories=50)
                _engine.attach_memory(_memory)
                _modes = CognitiveModes(n_oscillators=8, n_modes=4)
                _modes.label_modes(['focused', 'creative', 'exploratory', 'procedural'])
                # Fit modes with random phases initially.
                _modes.fit(torch.rand(50, 8) * 6.28)
                from fractus.train.online import OnlineTrainer
                _trainer = OnlineTrainer(_engine, lr=1e-3, accumulation_steps=4)
    return _engine


def _engine_state() -> dict:
    """Read ALL internal state from the engine."""
    eng = get_engine()
    ts = eng.thought_state
    return {
        "thought_norm": ts.norm().item(),
        "thought_mean": ts.mean().item(),
        "thought_std": ts.std().item(),
        "confidence": torch.sigmoid(eng.confidence_head(ts[:, 0, :])).item(),
        "lb_loss": eng.last_lb_loss.item() if hasattr(eng, 'last_lb_loss') else 0,
        "salience_loss": getattr(eng, 'last_salience_loss', torch.tensor(0)).item(),
        "tick_count": getattr(eng, '_tick_count', 0),
        "pert_max": getattr(eng, '_pert_max', 1.0),
        "n_blocks": len(eng.blocks),
        "n_experts": eng.blocks[0].moe.n_experts,
        "top_k": eng.blocks[0].moe.top_k,
        "memory_active": eng.memory_active,
        "memory_count": len(_memory) if _memory else 0,
    }


def _phases_state() -> list:
    """Phases Kuramoto par block."""
    eng = get_engine()
    out = []
    for i, blk in enumerate(eng.blocks):
        p = blk.kuramoto_phases[0, 0].tolist()
        out.append({"block": i, "phases": p})
    return out


def _routing_state() -> dict:
    """Expert routing distribution."""
    eng = get_engine()
    hits = getattr(eng, '_expert_hits', torch.zeros(eng.blocks[0].moe.n_experts))
    total = hits.sum().item()
    return {
        "expert_hits": hits.tolist(),
        "total": total,
        "dominant": hits.argmax().item() if total > 0 else -1,
        "dominance": (hits.max().item() / max(total, 1)),
        "n_experts": eng.blocks[0].moe.n_experts,
    }


def _mode_state() -> dict:
    """Current cognitive mode."""
    eng = get_engine()
    phases = eng.blocks[-1].kuramoto_phases[0, 0, :]  # (n_osc,)
    return _modes.classify(phases)


def _memory_state() -> list:
    """All persistent memories."""
    if not _memory or not _memory.vectors:
        return []
    return [{
        "idx": i,
        "context": _memory.contexts[i][:80],
        "importance": round(_memory.importance[i], 3),
        "vector_norm": _memory.vectors[i].norm().item(),
    } for i in range(len(_memory))]


def _training_state() -> dict:
    """Training metrics."""
    global _losses
    recent = _losses[-50:] if _losses else []
    avg = sum(recent) / max(len(recent), 1)
    return {
        "losses": recent,
        "avg_loss": avg,
        "ppl": math.exp(min(avg, 20)),
        "total_ticks": len(_losses),
        "trainer_losses": _trainer.losses[-20:] if _trainer and hasattr(_trainer, 'losses') else [],
    }


async def _broadcast_state():
    """Push state to all WebSocket clients."""
    if not _ws_clients:
        return
    state = {
        "type": "state",
        "engine": _engine_state(),
        "phases": _phases_state(),
        "routing": _routing_state(),
        "mode": _mode_state(),
        "time": time.time(),
    }
    msg = json.dumps(state)
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ─── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    # Send initial state.
    state = {"type": "state", "engine": _engine_state(), "phases": _phases_state(),
             "routing": _routing_state(), "mode": _mode_state(), "time": time.time()}
    await ws.send_text(json.dumps(state))
    try:
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        _ws_clients.discard(ws)


# ─── Monitoring endpoints ───────────────────────────────────────────────────

@app.get("/api/state")
async def api_state():
    return _engine_state()

@app.get("/api/phases")
async def api_phases():
    return _phases_state()

@app.get("/api/routing")
async def api_routing():
    return _routing_state()

@app.get("/api/mode")
async def api_mode():
    return _mode_state()

@app.get("/api/memory")
async def api_memory():
    return {"memories": _memory_state(), "count": len(_memory_state())}

@app.get("/api/training")
async def api_training():
    return _training_state()

@app.get("/api/params")
async def api_params():
    eng = get_engine()
    return {
        "d_model": eng.d_model,
        "n_layers": len(eng.blocks),
        "n_experts": eng.blocks[0].moe.n_experts,
        "top_k": eng.blocks[0].moe.top_k,
        "expert_rank": eng.blocks[0].moe.expert_rank,
        "n_oscillators": eng.blocks[0].kuramoto.N,
        "n_heads": eng.blocks[0].attn.n_heads,
        "d_head": eng.blocks[0].attn.d_head,
        "vocab_size": eng.vocab_size,
        "params": sum(p.numel() for p in eng.parameters()),
    }


# ─── Control endpoints ───────────────────────────────────────────────────────

class ChatReq(BaseModel):
    message: str

class InjectReq(BaseModel):
    text: str

class TrainReq(BaseModel):
    text: str
    steps: int = 10

class ConfigReq(BaseModel):
    memory_active: Optional[bool] = None
    blend: Optional[float] = None
    salience_bias: Optional[float] = None


@app.post("/api/chat")
async def api_chat(req: ChatReq):
    eng = get_engine()
    with _lock:
        ids = _tokenizer.encode(req.message)
        with torch.no_grad():
            for token_id in ids:
                logits, conf = eng.tick(torch.tensor([token_id]))
                _losses.append(F.cross_entropy(logits, torch.tensor([token_id])).item())
        # Generate response (greedy, up to 50 tokens).
        last = ids[-1] if ids else 0
        gen = list(ids)
        for _ in range(50):
            with torch.no_grad():
                logits, conf = eng.tick(torch.tensor([last]))
            nxt = logits.argmax(-1).item()
            if nxt == 50256:  # EOS
                break
            gen.append(nxt)
            last = nxt
        reply = _tokenizer.decode(gen)
        _tick_log.append({"input": req.message, "output": reply, "time": time.time()})
    await _broadcast_state()
    return {"reply": reply, "state": _engine_state(), "routing": _routing_state(),
            "mode": _mode_state()}


@app.post("/api/reset")
async def api_reset():
    eng = get_engine()
    with _lock:
        eng.reset_thought(batch_size=1)
    await _broadcast_state()
    return {"status": "reset", "state": _engine_state()}


@app.post("/api/tick")
async def api_tick():
    """One tick with no observation (pure thinking)."""
    eng = get_engine()
    with _lock:
        with torch.no_grad():
            logits, conf = eng.tick(None)
    await _broadcast_state()
    return {"confidence": conf.item(), "top_token": logits.argmax(-1).item(),
            "state": _engine_state()}


@app.post("/api/inject")
async def api_inject(req: InjectReq):
    """Inject text as tokens into the thought state."""
    eng = get_engine()
    with _lock:
        ids = _tokenizer.encode(req.text)
        with torch.no_grad():
            for tid in ids:
                eng.tick(torch.tensor([tid]))
    await _broadcast_state()
    return {"status": "injected", "n_tokens": len(ids), "state": _engine_state()}


@app.post("/api/memory/delete/{idx}")
async def api_memory_delete(idx: int):
    if _memory and 0 <= idx < len(_memory.vectors):
        with _lock:
            _memory.vectors.pop(idx)
            _memory.contexts.pop(idx)
            _memory.importance.pop(idx)
        return {"status": "deleted", "idx": idx, "remaining": len(_memory)}
    return {"error": "invalid index"}


@app.post("/api/memory/add")
async def api_memory_add(req: InjectReq):
    """Add a manual memory from text."""
    eng = get_engine()
    with _lock:
        ids = _tokenizer.encode(req.text)
        with torch.no_grad():
            h = eng.observe(torch.tensor(ids[-1:]))
        _memory.consolidate(h.flatten(), context=req.text, importance=0.8)
    return {"status": "added", "count": len(_memory)}


@app.post("/api/train")
async def api_train(req: TrainReq):
    """Train the engine on provided text (correction by gradient)."""
    eng = get_engine()
    with _lock:
        ids = _tokenizer.encode(req.text)
        if len(ids) < 4:
            return {"error": "need at least 4 tokens"}
        tokens = torch.tensor(ids)
        opt = torch.optim.SGD(eng.parameters(), lr=1e-3)
        chunk_len = min(16, max(4, len(ids) - 1))
        done = 0
        for step in range(req.steps):
            # Sliding window over the text; predict the token right after each chunk.
            start = (step * chunk_len) % max(len(tokens) - chunk_len, 1)
            chunk = tokens[start:start + chunk_len].unsqueeze(0)
            target_idx = min(start + chunk_len, len(tokens) - 1)
            target = tokens[target_idx:target_idx + 1]
            last_logits = eng.tick_chunk_train(chunk)
            loss = F.cross_entropy(last_logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
            opt.step()
            _losses.append(loss.item())
            done += 1
    await _broadcast_state()
    return {"status": "trained", "steps": done,
            "final_loss": _losses[-1] if _losses else 0,
            "training": _training_state()}


@app.post("/api/grow")
async def api_grow():
    eng = get_engine()
    with _lock:
        grew = eng.maybe_grow(max_experts=16, imbalance_threshold=0.5,
                              min_ticks_between_grows=1)
    await _broadcast_state()
    return {"grew": grew, "n_experts": eng.blocks[0].moe.n_experts,
            "routing": _routing_state()}


@app.post("/api/config")
async def api_config(req: ConfigReq):
    eng = get_engine()
    changes = {}
    if req.memory_active is not None:
        eng.memory_active = req.memory_active
        changes["memory_active"] = req.memory_active
    if req.salience_bias is not None:
        with torch.no_grad():
            eng.salience_head.bias.fill_(req.salience_bias)
        changes["salience_bias"] = req.salience_bias
    return {"status": "updated", "changes": changes, "state": _engine_state()}


# ─── Index ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    idx = STATIC / "index.html"
    if idx.exists():
        return idx.read_text(encoding="utf-8")
    return "<h1>Fractus Sandbox</h1><p>static/index.html missing</p>"


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": _engine is not None, "time": time.time()}
