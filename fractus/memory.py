"""PersistentMemory: long-term memory that survives across sessions.

THE INNOVATION. Claude and GPT forget everything between conversations.
This module gives the Continuous Thought Engine TRUE long-term memory:

    - A bank of "memory vectors" (d_model dimensional) stored on disk.
    - At startup, the engine loads its memories and injects them into the
      thought state — it "remembers" past interactions.
    - During operation, salient thoughts are periodically written back to
      the memory bank — the engine "learns" from experience.
    - Memories are keyed by context (what was happening when the memory
      formed), enabling associative recall.

This is the module that makes Fractus PERSONAL — it adapts to the user,
remembers preferences, and accumulates knowledge over time. No datacenter
needed; the memory lives on the user's machine.

Usage:
    memory = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
    engine.reset_thought()
    engine.inject_memory(memory)  # remember past context
    # ... think ...
    memory.consolidate(engine.thought_state, context="user asked about sorting")
    memory.save()
"""

import os

import torch
import torch.nn as nn


class PersistentMemory:
    """A persistent bank of memory vectors.

    Stores N memory slots, each (d_model,) + a text context label.
    Memories are recalled via cosine similarity to the current thought state.

    Args:
        d_model: dimension of memory vectors (must match the engine).
        max_memories: maximum number of stored memories (LRU eviction).
        path: file path for persistence (load/save).
    """

    def __init__(
        self,
        d_model: int = 128,
        max_memories: int = 256,
        path: str = None,
    ):
        self.d_model = d_model
        self.max_memories = max_memories
        self.path = path

        # Memory bank: vectors and their context labels.
        self.vectors = []   # list of (d_model,) tensors
        self.contexts = []  # list of strings
        self.importance = []  # list of floats (higher = more salient)

        # Load from disk if available.
        if path and os.path.exists(path):
            self.load()

    def recall(self, query: torch.Tensor, top_k: int = 3) -> list:
        """Recall the top-k most relevant memories for a query.

        Args:
            query: (d_model,) the current thought state.
            top_k: number of memories to recall.
        Returns:
            list of (context_label, similarity_score, vector) tuples.
        """
        if not self.vectors:
            return []

        # Stack all memories and compute cosine similarity.
        bank = torch.stack(self.vectors)  # (N, d_model)
        query_flat = query.flatten()  # (d_model,)

        # Cosine similarity.
        sims = torch.nn.functional.cosine_similarity(
            query_flat.unsqueeze(0), bank, dim=-1
        )  # (N,)

        # Top-k.
        k = min(top_k, len(self.vectors))
        topk_sims, topk_idx = sims.topk(k)
        results = []
        for i in range(k):
            idx = topk_idx[i].item()
            results.append((
                self.contexts[idx],
                topk_sims[i].item(),
                self.vectors[idx],
            ))
        return results

    def consolidate(
        self,
        thought_state: torch.Tensor,
        context: str = "",
        importance: float = 0.5,
    ):
        """Write a new memory from the current thought state.

        Args:
            thought_state: (1, 1, d_model) or (d_model,) the thought to remember.
            context: a text label describing when/why this memory formed.
            importance: salience score (higher = more likely to persist).
        """
        vec = thought_state.flatten().detach().cpu()
        if vec.shape[0] != self.d_model:
            return  # dimension mismatch, skip.

        self.vectors.append(vec)
        self.contexts.append(context)
        self.importance.append(importance)

        # LRU eviction: if over capacity, remove the least important memory.
        if len(self.vectors) > self.max_memories:
            min_idx = self.importance.index(min(self.importance))
            self.vectors.pop(min_idx)
            self.contexts.pop(min_idx)
            self.importance.pop(min_idx)

    def consolidate_if_salient(
        self,
        thought_state: torch.Tensor,
        salience_score: float,
        *,
        context: str = "",
        importance: float = 0.5,
        threshold: float = 0.7,
        min_distance: float = 0.1,
    ) -> bool:
        """Consolidate only if salient enough AND not a near-duplicate.

        This gates automatic consolidation by the CTE's salience head: a
        thought is only written to the memory bank when its salience score
        clears ``threshold`` and it is not a near-duplicate (cosine >
        ``1 - min_distance``) of any existing memory.

        Returns:
            True if a memory was written, False otherwise.
        """
        if salience_score < threshold:
            return False
        vec = thought_state.flatten().detach().cpu()
        if vec.shape[0] != self.d_model:
            return False
        # De-duplication: skip if too close to an existing memory.
        if self.vectors:
            bank = torch.stack(self.vectors)
            sim = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), bank, dim=-1)
            if sim.max().item() > 1.0 - min_distance:
                return False
        self.consolidate(vec, context=context, importance=importance)
        return True

    def inject(self, engine, top_k: int = 3, blend: float = 0.05) -> float:
        """Inject recalled memories into the engine's thought state.

        This is how the engine 'remembers' — past memories are added to
        the current thought, biasing it toward relevant context.

        Returns:
            The L2 norm of the perturbation caused by the injection
            (``||h_after - h_before||``). Zero if no memories were recalled.
            This is the intrinsic salience signal for the salience head: a
            large perturbation means the memory was relevant to the current
            thought state.

        Args:
            engine: an object with a (B, 1, d_model) ``thought_state``.
            top_k: number of memories to recall.
            blend: fraction of the memory contribution blended in.
                Default 0.05 (95% current thought + 5% memory) is tuned for
                continuous per-tick injection rather than one-shot resets.
        """
        if not self.vectors:
            return 0.0

        thought = engine.thought_state.flatten()  # (d_model,)
        recalled = self.recall(thought, top_k=top_k)

        if recalled:
            # Weighted sum of recalled memories, added to the thought.
            total_weight = 0.0
            memory_contribution = torch.zeros_like(thought)
            for ctx, sim, vec in recalled:
                weight = max(sim, 0.0)  # only positive correlations
                memory_contribution += weight * vec
                total_weight += weight
            if total_weight > 0:
                memory_contribution /= total_weight
                # Measure the perturbation BEFORE applying it.
                # Δ = blend * memory_contribution (the additive part that shifts the thought).
                delta = blend * memory_contribution.to(engine.thought_state.device)
                perturbation = delta.norm().item()  # L2 norm of the shift
                # Blend: (1-blend) current thought + blend memory.
                engine.thought_state[:, 0, :] = (
                    (1.0 - blend) * engine.thought_state[:, 0, :] +
                    blend * memory_contribution.to(engine.thought_state.device)
                )
                return perturbation
        return 0.0

    def save(self, path: str = None):
        """Save the memory bank to disk."""
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "vectors": [v.tolist() for v in self.vectors],
            "contexts": self.contexts,
            "importance": self.importance,
            "d_model": self.d_model,
        }
        torch.save(data, path)

    def load(self, path: str = None):
        """Load the memory bank from disk."""
        path = path or self.path
        if not path or not os.path.exists(path):
            return
        data = torch.load(path, weights_only=False)
        self.d_model = data.get("d_model", self.d_model)
        self.vectors = [torch.tensor(v, dtype=torch.float32) for v in data["vectors"]]
        self.contexts = data["contexts"]
        self.importance = data["importance"]

    def clear(self):
        """Wipe all memories (factory reset)."""
        self.vectors = []
        self.contexts = []
        self.importance = []

    def __len__(self):
        return len(self.vectors)

    def summary(self) -> str:
        """Human-readable summary of stored memories."""
        if not self.vectors:
            return "Memory bank: empty"
        lines = [f"Memory bank: {len(self.vectors)} memories"]
        for i, (ctx, imp) in enumerate(zip(self.contexts, self.importance)):
            lines.append(f"  [{i}] imp={imp:.2f}  {ctx[:60]}")
        return "\n".join(lines)
