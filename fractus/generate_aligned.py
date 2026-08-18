"""Train-aligned generation for Fractus CTE.

Uses tick_chunk only (same path as stage2 training), never tick_single.
"""
from __future__ import annotations
import torch
from typing import List, Optional

@torch.no_grad()
def generate_chunk(
    engine,
    tokenizer,
    prompt: str,
    max_new: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
    ban_window: int = 8,
    ban_factor: float = 0.4,
    context_limit: int = 128,
) -> tuple[str, List[int]]:
    engine.eval()
    engine.reset_thought(1)
    for blk in engine.blocks:
        if hasattr(blk, 'attn_S'):
            blk.attn_S.zero_()
        if hasattr(blk, 'attn_z'):
            blk.attn_z.zero_()

    ids = tokenizer.encode(prompt)[:context_limit]
    if not ids:
        ids = [0]

    # warm full prompt as one chunk
    logits = engine.tick_chunk(torch.tensor([ids], dtype=torch.long))
    cur = logits[0, -1]
    out: List[int] = []

    for _ in range(max_new):
        l = cur.float() / max(temperature, 1e-5)
        for prev in set(out[-ban_window:]):
            l[prev] *= ban_factor
        k = min(top_k, l.size(-1))
        topv, topi = torch.topk(l, k)
        nxt = int(topi[torch.multinomial(torch.softmax(topv, -1), 1)].item())
        out.append(nxt)
        # advance with train path (length-1 chunk)
        logits = engine.tick_chunk(torch.tensor([[nxt]], dtype=torch.long))
        cur = logits[0, -1]

    return tokenizer.decode(out), out


@torch.no_grad()
def generate_window(
    engine,
    tokenizer,
    prompt: str,
    max_new: int = 40,
    temperature: float = 0.8,
    top_k: int = 40,
    window: int = 64,
    ban_window: int = 8,
    ban_factor: float = 0.4,
) -> tuple[str, List[int]]:
    """Re-encode last  tokens each step (fresh causal context)."""
    engine.eval()
    ids = tokenizer.encode(prompt)[:window]
    out: List[int] = []
    for _ in range(max_new):
        engine.reset_thought(1)
        for blk in engine.blocks:
            if hasattr(blk, 'attn_S'):
                blk.attn_S.zero_()
            if hasattr(blk, 'attn_z'):
                blk.attn_z.zero_()
        ctx = ids[-window:]
        logits = engine.tick_chunk(torch.tensor([ctx], dtype=torch.long))
        l = logits[0, -1].float() / max(temperature, 1e-5)
        for prev in set(out[-ban_window:]):
            l[prev] *= ban_factor
        topv, topi = torch.topk(l, min(top_k, l.size(-1)))
        nxt = int(topi[torch.multinomial(torch.softmax(topv, -1), 1)].item())
        out.append(nxt)
        ids.append(nxt)
    return tokenizer.decode(out), out
