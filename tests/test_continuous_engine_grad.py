"""Regression test for the frozen-expert Phase 3 defect.

Before the fix, tick_chunk read the experts' detached _cached_W buffer
directly, so expert params received no gradient. This test fails on the
old code and must pass on the refactored engine.

Also tests multi-block gradient flow and structure.
"""
import torch
from fractus.continuous_engine import ContinuousThoughtEngine


def _build_13m(n_layers=1):
    return ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_layers=n_layers, n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )


def test_cte_experts_receive_gradient():
    """Every expert parameter gets a finite, non-zero gradient after tick_chunk + backward."""
    eng = _build_13m()
    eng.reset_thought(batch_size=1)
    tokens = torch.randint(0, eng.vocab_size, (1, 16))
    logits = eng.tick_chunk(tokens)
    loss = logits.pow(2).mean()
    loss.backward()

    # MoE is at eng.blocks[0].moe.
    moe_params = list(eng.blocks[0].moe.named_parameters())
    assert len(moe_params) > 0, "blocks[0].moe has no parameters"
    for name, p in moe_params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received NO gradient (frozen-expert bug)"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"
        assert p.grad.abs().sum().item() > 0, f"{name} received zero gradient"


def test_cte_has_moe_attribute():
    """The CTE must use PhaseRoutedMoE in each block."""
    from fractus.nn.moe import PhaseRoutedMoE
    eng = _build_13m()
    assert isinstance(eng.blocks[0].moe, PhaseRoutedMoE)
    assert len(eng.blocks) == 1, "default should be 1 block"


def test_cte_multi_block_gradient_flow():
    """Multi-block CTE: gradient flows through ALL blocks."""
    eng = _build_13m(n_layers=3)
    assert len(eng.blocks) == 3
    eng.reset_thought(batch_size=1)
    tokens = torch.randint(0, eng.vocab_size, (1, 16))
    logits = eng.tick_chunk(tokens)
    loss = logits.pow(2).mean()
    loss.backward()

    # Every block's MoE must receive gradient.
    for i, blk in enumerate(eng.blocks):
        moe_params = list(blk.moe.parameters())
        assert len(moe_params) > 0, f"block {i} moe has no params"
        for p in moe_params:
            assert p.grad is not None, f"block {i} moe param received NO gradient"
            assert p.grad.abs().sum().item() > 0, f"block {i} moe param got zero grad"

    # Every block's attention must receive gradient.
    for i, blk in enumerate(eng.blocks):
        attn_params = list(blk.attn.parameters())
        for p in attn_params:
            assert p.grad is not None, f"block {i} attn param received NO gradient"


def test_cte_multi_block_continuous_thought():
    """Multi-block: thought_state and per-block (S,z) carry across chunks."""
    eng = _build_13m(n_layers=2)
    eng.reset_thought(batch_size=1)

    chunk1 = torch.tensor([[i + 1 for i in range(16)]])
    chunk2 = torch.tensor([[i + 17 for i in range(16)]])

    eng.tick_chunk(chunk1)
    ts1 = eng.thought_state.clone()
    S1_0 = eng.blocks[0].attn_S.clone()
    S1_1 = eng.blocks[1].attn_S.clone()

    eng.tick_chunk(chunk2)
    ts2 = eng.thought_state.clone()
    S2_0 = eng.blocks[0].attn_S.clone()
    S2_1 = eng.blocks[1].attn_S.clone()

    assert not torch.equal(ts1, ts2), "thought_state must change across chunks"
    assert not torch.equal(S1_0, S2_0), "block 0 S must change"
    assert not torch.equal(S1_1, S2_1), "block 1 S must change"
    assert S2_0.abs().max() > 0, "block 0 S must be nonzero"
    assert S2_1.abs().max() > 0, "block 1 S must be nonzero"

