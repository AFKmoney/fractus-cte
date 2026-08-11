"""Fractus Progressive Growth — grow a model's width, depth, and expert count.

THE INNOVATION. Instead of training a large model from scratch (impossible on
CPU), we grow it palier by palier. Each palier inherits the previous model's
weights via zero-padding (for width) or copying (for depth/experts), then
trains briefly. The model never starts from random — it starts "warm".

Growth axes:
  - WIDTH (d_model): zero-pad every d-coupled matrix to the new dimension.
  - DEPTH (n_layers): copy old blocks, init new ones with standard scheme.
  - EXPERTS (n_experts): copy old experts, zero-init new ones.
  - RANK (siren_rank): zero-pad the rank dimension of U/V factors.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict, Any


def _pad_dim0(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """Grow a tensor along dim 0, zero-padding the new rows."""
    old = tensor.shape[0]
    if old >= new_size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[0] = new_size - old
    pad = torch.zeros(pad_shape, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


def _pad_last_dim(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """Grow a tensor along the LAST dim, zero-padding the new columns."""
    old = tensor.shape[-1]
    if old >= new_size:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[-1] = new_size - old
    pad = torch.zeros(pad_shape, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=-1)


def _transfer_block_weights(old_blk, new_blk, old_d: int, new_d: int):
    """Transfer weights from old CTEBlock to new CTEBlock via zero-padding.

    Copies old knowledge into the top-left block of every matrix. New dims
    are zero (neutral start). New experts/blocks warm up during training.
    """

    # 1. Attention w_qkv: list of 3 tensors, each (d_model, d_model).
    if hasattr(old_blk.attn, "w_qkv"):
        for i in range(min(len(old_blk.attn.w_qkv), len(new_blk.attn.w_qkv))):
            old_w = old_blk.attn.w_qkv[i].data
            new_w = new_blk.attn.w_qkv[i].data
            r_copy = min(old_w.shape[0], new_w.shape[0])
            c_copy = min(old_w.shape[1], new_w.shape[1])
            new_w.zero_()
            new_w[:r_copy, :c_copy] = old_w[:r_copy, :c_copy]
    # Attention biases.
    if hasattr(old_blk.attn, "b_qkv"):
        for i in range(min(len(old_blk.attn.b_qkv), len(new_blk.attn.b_qkv))):
            old_b = old_blk.attn.b_qkv[i].data
            new_b = new_blk.attn.b_qkv[i].data
            c_copy = min(old_b.shape[0], new_b.shape[0])
            new_b[:c_copy] = old_b[:c_copy]
    # Attention w_out.
    if hasattr(old_blk.attn, "w_out"):
        old_wo = old_blk.attn.w_out.data
        new_wo = new_blk.attn.w_out.data
        r_copy = min(old_wo.shape[0], new_wo.shape[0])
        c_copy = min(old_wo.shape[1], new_wo.shape[1])
        new_wo.zero_()
        new_wo[:r_copy, :c_copy] = old_wo[:r_copy, :c_copy]
    if hasattr(old_blk.attn, "b_out"):
        old_bo = old_blk.attn.b_out.data
        new_bo = new_blk.attn.b_out.data
        c_copy = min(old_bo.shape[0], new_bo.shape[0])
        new_bo[:c_copy] = old_bo[:c_copy]
    # Level offsets.
    if hasattr(old_blk.attn, "level_offsets") and hasattr(new_blk.attn, "level_offsets"):
        old_lo = old_blk.attn.level_offsets.data
        new_lo = new_blk.attn.level_offsets.data
        n_copy = min(old_lo.shape[0], new_lo.shape[0])
        new_lo[:n_copy] = old_lo[:n_copy]
    # Level logits.
    if hasattr(old_blk.attn, "level_logits") and hasattr(new_blk.attn, "level_logits"):
        old_ll = old_blk.attn.level_logits.data
        new_ll = new_blk.attn.level_logits.data
        n_copy = min(old_ll.shape[0], new_ll.shape[0])
        new_ll[:n_copy] = old_ll[:n_copy]

    # 2. LayerNorms: copy old dims, gamma=1/beta=0 for new.
    for (old_norm, new_norm) in [
        (old_blk.norm_attn, new_blk.norm_attn),
        (old_blk.norm_kur, new_blk.norm_kur),
        (old_blk.norm_moe, new_blk.norm_moe),
    ]:
        old_g = old_norm.weight.data
        new_g = new_norm.weight.data
        d_copy = min(old_g.shape[0], new_g.shape[0])
        new_g[:d_copy] = old_g[:d_copy]
        old_b = old_norm.bias.data
        new_b = new_norm.bias.data
        new_b[:d_copy] = old_b[:d_copy]

    # 3. Kuramoto: grow oscillators + coupling rank.
    if hasattr(old_blk.kuramoto, "omega"):
        old_om = old_blk.kuramoto.omega.data
        new_om = new_blk.kuramoto.omega.data
        n_copy = min(old_om.shape[0], new_om.shape[0])
        new_om[:n_copy] = old_om[:n_copy]
    if hasattr(old_blk.kuramoto, "coupling_u"):
        old_cu = old_blk.kuramoto.coupling_u.data
        new_cu = new_blk.kuramoto.coupling_u.data
        n_copy = min(old_cu.shape[0], new_cu.shape[0])
        r_copy = min(old_cu.shape[1], new_cu.shape[1])
        new_cu[:n_copy, :r_copy] = old_cu[:n_copy, :r_copy]
    if hasattr(old_blk.kuramoto, "coupling_lambda"):
        old_cl = old_blk.kuramoto.coupling_lambda.data
        new_cl = new_blk.kuramoto.coupling_lambda.data
        r_copy = min(old_cl.shape[0], new_cl.shape[0])
        new_cl[:r_copy] = old_cl[:r_copy]

    # 4. MoE experts (low-rank): U1/V1/U2/V2 are (E, ..., r).
    moe_old = old_blk.moe
    moe_new = new_blk.moe
    e_copy = min(moe_old.n_experts, moe_new.n_experts)

    if moe_old.expert_rank is not None and moe_new.expert_rank is not None:
        for param_name in ["U1", "V1", "U2", "V2", "scale1", "scale2", "b1", "b2"]:
            old_p = getattr(moe_old, param_name).data
            new_p = getattr(moe_new, param_name).data
            new_p.zero_()  # neutral start for ALL
            if old_p.dim() == 3:
                dd = min(old_p.shape[1], new_p.shape[1])
                rr = min(old_p.shape[2], new_p.shape[2])
                new_p[:e_copy, :dd, :rr] = old_p[:e_copy, :dd, :rr]
            elif old_p.dim() == 2:
                dd = min(old_p.shape[1], new_p.shape[1])
                new_p[:e_copy, :dd] = old_p[:e_copy, :dd]
            elif old_p.dim() == 1:
                new_p[:e_copy] = old_p[:e_copy]
            # Restore scale=1 for OLD experts.
            if param_name in ("scale1", "scale2"):
                old_scale = getattr(moe_old, param_name).data[:e_copy]
                new_p[:e_copy] = old_scale


def grow_cte(old_engine, new_config: Dict[str, Any]):
    """Grow a ContinuousThoughtEngine to a larger config.

    Copies old weights into the new model via zero-padding. Supports growth
    in width (d_model), depth (n_layers), experts (n_experts), and rank
    (siren_rank). Old knowledge is preserved; new capacity is neutral.

    Args:
        old_engine: a trained ContinuousThoughtEngine.
        new_config: dict with any of:
            d_model, n_layers, n_experts, top_k, expert_d_ff, siren_rank,
            n_heads, d_head, n_levels, n_oscillators, coupling_rank, vocab_size.
    """
    from .continuous_engine import ContinuousThoughtEngine

    old_d = old_engine.d_model
    old_vocab = old_engine.vocab_size
    old_n_layers = len(old_engine.blocks)

    new_d = new_config.get("d_model", old_d)
    new_vocab = new_config.get("vocab_size", old_vocab)
    new_n_layers = new_config.get("n_layers", old_n_layers)
    new_n_experts = new_config.get("n_experts", old_engine.blocks[0].moe.n_experts)
    new_rank = new_config.get("siren_rank", old_engine.blocks[0].moe.expert_rank or 32)
    new_d_ff = new_config.get("expert_d_ff", old_engine.blocks[0].moe.d_ff)
    new_top_k = new_config.get("top_k", old_engine.blocks[0].moe.top_k)
    new_n_heads = new_config.get("n_heads", old_engine.blocks[0].attn.n_heads)
    new_d_head = new_config.get("d_head", old_engine.blocks[0].attn.d_head)
    new_n_levels = new_config.get("n_levels", old_engine.blocks[0].attn.n_levels)
    new_n_osc = new_config.get("n_oscillators", old_engine.blocks[0].kuramoto.N)
    new_coupling_rank = new_config.get("coupling_rank", old_engine.blocks[0].kuramoto.rank)

    # Build the new engine.
    new_engine = ContinuousThoughtEngine(
        vocab_size=new_vocab, d_model=new_d,
        n_heads=new_n_heads, d_head=new_d_head, n_levels=new_n_levels,
        n_oscillators=new_n_osc, coupling_rank=new_coupling_rank,
        n_experts=new_n_experts, top_k=new_top_k,
        expert_d_ff=new_d_ff, siren_rank=(new_rank if new_rank else None),
        n_layers=new_n_layers,
    )

    # --- Transfer embedding (shared across all blocks) ---
    old_emb = old_engine.observe.weight.data
    new_emb = new_engine.observe.weight.data
    v_copy = min(old_vocab, new_vocab)
    d_copy = min(old_d, new_d)
    new_emb.zero_()
    new_emb[:v_copy, :d_copy] = old_emb[:v_copy, :d_copy]

    # --- Transfer per-block weights (loop over ALL old blocks) ---
    blocks_to_copy = min(old_n_layers, new_n_layers)
    for blk_idx in range(blocks_to_copy):
        _transfer_block_weights(
            old_engine.blocks[blk_idx],
            new_engine.blocks[blk_idx],
            old_d, new_d,
        )
    # New blocks (indices old_n_layers..new_n_layers-1) keep random init —
    # they warm up during training (like new brain regions developing).

    # --- Transfer heads (top-level, not per-block) ---
    for head_name in ["confidence_head", "salience_head"]:
        old_h = getattr(old_engine, head_name)
        new_h = getattr(new_engine, head_name)
        old_w = old_h.weight.data
        new_w = new_h.weight.data
        d_copy = min(old_w.shape[1], new_w.shape[1])
        new_w[:, :d_copy] = old_w[:, :d_copy]
        new_h.bias.data[:] = old_h.bias.data[:]

    # Output head is TIED with embedding — already handled above.

    return new_engine


def grow_summary(old_engine, new_engine) -> dict:
    """Report what changed between old and new engine."""
    return {
        "d_model": f"{old_engine.d_model} → {new_engine.d_model}",
        "n_layers": f"{len(old_engine.blocks)} → {len(new_engine.blocks)}",
        "n_experts": f"{old_engine.blocks[0].moe.n_experts} → {new_engine.blocks[0].moe.n_experts}",
        "expert_rank": f"{old_engine.blocks[0].moe.expert_rank} → {new_engine.blocks[0].moe.expert_rank}",
        "params": f"{sum(p.numel() for p in old_engine.parameters()):,} → {sum(p.numel() for p in new_engine.parameters()):,}",
        "n_oscillators": f"{old_engine.blocks[0].kuramoto.N} → {new_engine.blocks[0].kuramoto.N}",
    }
