"""Tests for PersistentMemory (isolated from the engine)."""
import tempfile, os
import torch
import pytest
from fractus.memory import PersistentMemory


def test_memory_recall_topk():
    mem = PersistentMemory(d_model=5, max_memories=10)
    # Store 5 vectors; query should rank the closest first.
    for i in range(5):
        v = torch.zeros(5); v[i] = 1.0
        mem.consolidate(v, context=f"mem{i}", importance=0.5)
    query = torch.zeros(5); query[0] = 1.0  # identical to mem0
    results = mem.recall(query, top_k=3)
    assert len(results) == 3
    assert results[0][0] == "mem0"  # highest cosine
    assert results[0][1] > results[1][1]  # descending


def test_memory_consolidate_lru():
    mem = PersistentMemory(d_model=4, max_memories=3)
    for i in range(5):
        mem.consolidate(torch.randn(4), context=f"mem{i}", importance=float(i) / 10)
    assert len(mem) == 3
    # Lowest importance (mem0, imp=0.0) should have been evicted.
    contexts = mem.contexts
    assert "mem0" not in contexts


def test_memory_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.pt")
        mem = PersistentMemory(d_model=4, max_memories=10, path=path)
        mem.consolidate(torch.tensor([1.0, 0, 0, 0]), context="hello", importance=0.8)
        mem.save()
        mem2 = PersistentMemory(d_model=4, max_memories=10, path=path)
        assert len(mem2) == 1
        assert mem2.contexts[0] == "hello"
        assert torch.allclose(mem2.vectors[0], torch.tensor([1.0, 0, 0, 0]))


def test_memory_inject_blend_ratio():
    """inject with blend=0.05 → thought_state is 95% original + 5% memory."""
    mem = PersistentMemory(d_model=4, max_memories=10)
    mem.consolidate(torch.tensor([0.0, 0.0, 0.0, 10.0]), context="big", importance=1.0)

    class FakeEngine:
        def __init__(self):
            self.thought_state = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])  # (1,1,4)

    eng = FakeEngine()
    before = eng.thought_state.clone()
    mem.inject(eng, blend=0.05, top_k=1)
    after = eng.thought_state
    # The memory vector is [0,0,0,10]; recalled contribution pushes toward it.
    # dim 3 (index 3) should increase strongly, while dims 0-2 should move only
    # by the (1-blend) shrinkage (~0.05 at blend=0.05), since the memory is 0
    # there. The blend is convex: new = (1-blend)*thought + blend*memory.
    assert after[0, 0, 3] > before[0, 0, 3]  # memory's non-zero dim increased
    # dims 0-2 are barely affected (small shrinkage, not a memory-driven push)
    assert abs(after[0, 0, 0].item() - before[0, 0, 0].item()) < 0.1


def test_memory_consolidate_if_salient_threshold():
    mem = PersistentMemory(d_model=4, max_memories=10)
    v = torch.tensor([1.0, 0, 0, 0])
    # Below threshold → no write.
    wrote = mem.consolidate_if_salient(v, salience_score=0.3, context="low")
    assert wrote is False
    assert len(mem) == 0
    # Above threshold → write.
    wrote = mem.consolidate_if_salient(v, salience_score=0.9, context="high")
    assert wrote is True
    assert len(mem) == 1


def test_memory_consolidate_if_salient_dedup():
    mem = PersistentMemory(d_model=4, max_memories=10)
    v = torch.tensor([1.0, 0, 0, 0])
    mem.consolidate_if_salient(v, salience_score=0.9, context="first")
    # Same vector again (cosine ~1.0, within min_distance) → no write.
    wrote = mem.consolidate_if_salient(v, salience_score=0.9, context="dup",
                                       min_distance=0.1)
    assert wrote is False
    assert len(mem) == 1
    # Different vector (cosine < 1 - min_distance) → write.
    v2 = torch.tensor([0.0, 1.0, 0, 0])
    wrote = mem.consolidate_if_salient(v2, salience_score=0.9, context="new",
                                       min_distance=0.1)
    assert wrote is True
    assert len(mem) == 2


def test_cte_has_salience_head_and_memory_slot():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    assert hasattr(eng, "salience_head")
    assert hasattr(eng, "memory")
    assert eng.memory is None
    assert eng.memory_active is True


def test_cte_attach_detach_memory():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    assert eng.memory is mem
    eng.detach_memory()
    assert eng.memory is None


def test_cte_salience_head_outputs_in_range():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    # salience_head output is captured during tick; we check via a direct call.
    with torch.no_grad():
        score = torch.sigmoid(eng.salience_head(eng.thought_state[:, 0, :]))
    assert 0.0 <= score.item() <= 1.0


def test_cte_consolidates_when_salient():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    # Force salience_head to output ~1.0 (set bias high).
    with torch.no_grad():
        eng.salience_head.bias.fill_(10.0)
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    assert len(mem) > 0, "should have consolidated a salient thought"


def test_cte_skips_consolidate_when_not_salient():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    # Force salience_head to output ~0.0 (set bias very negative).
    with torch.no_grad():
        eng.salience_head.bias.fill_(-10.0)
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    assert len(mem) == 0, "should NOT have consolidated a non-salient thought"


def test_cte_memory_active_toggle():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    with torch.no_grad():
        eng.salience_head.bias.fill_(10.0)  # salient
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    eng.memory_active = False  # disabled
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    assert len(mem) == 0, "memory_active=False should prevent consolidation"
