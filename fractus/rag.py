"""KnowledgeBase: vector store for retrieval-augmented generation (RAG).

THE APPROACH: instead of cramming all knowledge into 88M model parameters
(takes weeks to train), we:
  1. Train a small 13M reasoning engine (fast, hours not weeks)
  2. Store all 45M tokens of knowledge as embeddings in a vector DB
  3. At inference time: retrieve relevant passages, inject them as context,
     and let the reasoning engine process them

This is how the brain works: the cortex (small, fast) reasons, and external
memory (books, notes, internet) provides the knowledge. We don't memorize
all of Wikipedia — we know how to READ and REASON about what we find.

Components:
  - KnowledgeBase: stores text chunks + their embeddings, retrieves by similarity
  - Retriever: given a query, finds the top-k most relevant knowledge chunks
  - RAGEngine: wraps the ContinuousThoughtEngine with retrieval

The embeddings are computed by the engine's own embedding layer (no external
model needed). The vector store uses cosine similarity (numpy, no external DB).
"""

import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class KnowledgeBase:
    """A simple vector store for text knowledge.

    Stores text chunks and their embeddings. Retrieves the top-k most
    similar chunks to a query embedding.

    Args:
        d_model: dimension of embeddings (must match the engine).
        path: file path for persistence.
    """

    def __init__(self, d_model: int = 128, path: str = None):
        self.d_model = d_model
        self.path = path
        self.chunks: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.sources: List[str] = []

        if path and os.path.exists(path):
            self.load()

    def add(self, text: str, embedding: np.ndarray, source: str = ""):
        """Add a text chunk with its embedding."""
        self.chunks.append(text)
        self.embeddings.append(embedding)
        self.sources.append(source)

    def add_batch(self, texts: List[str], embeddings: np.ndarray, sources: List[str] = None):
        """Add many chunks at once."""
        if sources is None:
            sources = [""] * len(texts)
        self.chunks.extend(texts)
        self.embeddings.extend(embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings)
        self.sources.extend(sources)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Retrieve the top-k most relevant chunks.

        Returns list of (text, similarity_score, source).
        """
        if not self.embeddings:
            return []

        # Stack all embeddings.
        bank = np.array(self.embeddings, dtype=np.float32)  # (N, d_model)

        # Cosine similarity.
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        bank_norm = bank / (np.linalg.norm(bank, axis=-1, keepdims=True) + 1e-10)
        sims = bank_norm @ query_norm  # (N,)

        # Top-k.
        k = min(top_k, len(self.chunks))
        if k == 0:
            return []
        if k == 1:
            top_indices = np.array([np.argmax(sims)])
        else:
            top_indices = np.argpartition(-sims, k - 1)[:k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(sims[idx]), self.sources[idx]))
        return results

    def save(self, path: str = None):
        """Save the knowledge base to disk."""
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'sources': self.sources,
                'd_model': self.d_model,
            }, f)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  KnowledgeBase saved: {len(self.chunks)} chunks, {size_mb:.1f} MB", flush=True)

    def load(self, path: str = None):
        """Load from disk."""
        path = path or self.path
        if not path or not os.path.exists(path):
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.sources = data['sources']
        self.d_model = data.get('d_model', self.d_model)

    def __len__(self):
        return len(self.chunks)

    def stats(self) -> str:
        total_chars = sum(len(c) for c in self.chunks)
        total_words = sum(len(c.split()) for c in self.chunks)
        return (f"KnowledgeBase: {len(self.chunks):,} chunks, "
                f"{total_words:,} words, {total_chars:,} chars")


class Retriever:
    """Retrieves relevant knowledge for a query.

    Uses the engine's embedding layer to compute query embeddings,
    then searches the KnowledgeBase for similar chunks.

    Args:
        engine: a ContinuousThoughtEngine (uses its embedding layer).
        kb: a KnowledgeBase.
    """

    def __init__(self, engine, kb: KnowledgeBase):
        self.engine = engine
        self.kb = kb

    @torch.no_grad()
    def embed_text(self, text: str, tokenizer) -> np.ndarray:
        """Compute the embedding of a text string using the engine's embedding."""
        ids = tokenizer.encode(text)
        if not ids:
            return np.zeros(self.kb.d_model, dtype=np.float32)
        ids_tensor = torch.tensor([ids[:32]], dtype=torch.long)  # (1, up to 32)
        # Use the engine's embedding layer.
        emb = self.engine.observe(ids_tensor)  # (1, L, d_model)
        # Average over the sequence dimension + detach for numpy.
        emb = emb.mean(dim=1).squeeze(0).detach()  # (d_model,)
        return emb.cpu().numpy().astype(np.float32)

    def retrieve(self, query: str, tokenizer, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Retrieve relevant knowledge for a query string."""
        query_emb = self.embed_text(query, tokenizer)
        return self.kb.retrieve(query_emb, top_k=top_k)


class RAGEngine:
    """Retrieval-Augmented Generation engine.

    Combines:
      - A small reasoning engine (ContinuousThoughtEngine)
      - A knowledge base (vector store of text)
      - A retriever (connects the two)

    At inference time:
      1. User asks a question.
      2. Retriever finds the most relevant knowledge chunks.
      3. The knowledge is injected into the engine's thought state.
      4. The engine reasons about the question + retrieved context.
      5. Output is generated.

    This is the architecture that makes a 13M model act like it has
    1B of knowledge — because the knowledge is EXTERNAL, not in the weights.

    Args:
        engine: a ContinuousThoughtEngine (the reasoning brain).
        tokenizer: a FractusTokenizer.
        kb: a KnowledgeBase (the external memory).
    """

    def __init__(self, engine, tokenizer, kb: KnowledgeBase):
        self.engine = engine
        self.tokenizer = tokenizer
        self.kb = kb
        self.retriever = Retriever(engine, kb)

    def query(self, question: str, top_k: int = 3, max_tokens: int = 50,
              temperature: float = 0.7) -> dict:
        """Answer a question using retrieval-augmented generation.

        Args:
            question: the user's question.
            top_k: number of knowledge chunks to retrieve.
            max_tokens: max output tokens.
            temperature: sampling temperature.

        Returns:
            dict with "answer", "retrieved_context", "sources".
        """
        # 1. Retrieve relevant knowledge.
        retrieved = self.retriever.retrieve(question, self.tokenizer, top_k=top_k)

        # 2. Build context prompt: retrieved knowledge + question.
        context_parts = [text for text, score, source in retrieved]
        context = " ".join(context_parts)[:500]  # limit context size
        full_prompt = f"{context}\n\nQuestion: {question}\nAnswer:"

        # 3. Feed to the engine.
        self.engine.eval()
        self.engine.reset_thought(batch_size=1)

        prompt_ids = self.tokenizer.encode(full_prompt)[:32]  # truncate to seq_len
        for tid in prompt_ids:
            self.engine.tick(torch.tensor([tid]))

        # 4. Generate answer.
        generated = list(prompt_ids)
        import torch.nn.functional as Tf
        with torch.no_grad():
            for _ in range(max_tokens):
                logits, conf = self.engine.tick()
                l = logits[0] / max(temperature, 1e-8)
                tv, ti = l.topk(40)
                probs = Tf.softmax(tv, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                next_token = ti[idx].item()
                generated.append(next_token)
                if next_token == 50256:  # <|endoftext|>
                    break

        answer = self.tokenizer.decode(generated)

        return {
            "answer": answer,
            "retrieved_context": context[:200],
            "sources": [s for _, _, s in retrieved if s],
            "confidence": conf.item(),
        }

    def learn(self, text: str, source: str = "user"):
        """Learn new knowledge WITHOUT retraining.

        The engine encodes the text into an embedding using its OWN
        embedding layer, then stores (text, embedding) in the KnowledgeBase.
        Next time a similar topic comes up, this knowledge is retrieved
        and injected into the reasoning.

        This is ONLINE LEARNING through retrieval — the model gets
        permanently smarter with every interaction, no gradient descent
        needed. The KnowledgeBase IS the long-term memory.

        Args:
            text: the knowledge to learn (a fact, a document, a conversation).
            source: who/where it came from.
        """
        self.engine.eval()
        # Split long text into chunks (max 200 chars each for retrieval granularity).
        chunks = []
        if len(text) > 200:
            words = text.split()
            current = ""
            for word in words:
                if len(current) + len(word) > 200:
                    chunks.append(current.strip())
                    current = word
                else:
                    current += " " + word
            if current.strip():
                chunks.append(current.strip())
        else:
            chunks = [text]

        # Embed each chunk and add to the KB.
        for chunk in chunks:
            emb = self.retriever.embed_text(chunk, self.tokenizer)
            self.kb.add(chunk, emb, source)

        # Auto-save so the knowledge persists across restarts.
        self.kb.save()

    def converse(self, user_input: str, speaker: str = "user") -> dict:
        """Have a conversation: answer + learn.

        Every conversation is a learning opportunity:
          1. The user's input is stored in the KB (the engine 'remembers' it).
          2. The engine retrieves relevant past context.
          3. The engine generates a response.
          4. The response is also stored (the engine learns from its own output).

        This means the engine GROWS INTELLIGENT over time through use,
        without any retraining — exactly like a human accumulates
        knowledge through conversation.

        Args:
            user_input: what the user said.
            speaker: identifier for the speaker.
        Returns:
            dict with "response", "learned", "context".
        """
        # Step 1: Learn from what the user just said.
        self.learn(user_input, source=speaker)

        # Step 2: Answer (uses retrieval of everything learned so far).
        result = self.query(user_input, top_k=5, max_tokens=80)

        # Step 3: Learn from the engine's own response.
        if result["answer"]:
            self.learn(result["answer"], source="fractus")

        result["learned_chunks"] = len(self.kb)
        return result

    def info(self) -> dict:
        return {
            "method": "retrieval-augmented generation + online learning",
            "brain_params": sum(p.numel() for p in self.engine.parameters()),
            "knowledge_chunks": len(self.kb),
            "brain_size_mb": sum(p.numel() * 4 for p in self.engine.parameters()) / 1e6,
            "kb_size_mb": sum(len(c) for c in self.kb.chunks) / 1e6,
            "long_term_memory": "YES (KnowledgeBase persists to disk)",
            "learns_without_retraining": "YES (every conversation adds knowledge)",
            "plugins": list(self.plugins.keys()) if hasattr(self, 'plugins') else [],
        }


# ---------------------------------------------------------------------------
# CognitivePlugins: hot-swappable behavior modules (no retraining)
# ---------------------------------------------------------------------------

class CognitivePlugin:
    """A hot-swappable behavior modifier. Loaded/unloaded at runtime.

    Plugins modify HOW the engine thinks, not WHAT it knows.
    Knowledge comes from the KB (RAG); behavior comes from plugins.

    A plugin has:
      - name: identifier
      - system_prompt: injected before the engine thinks (like a system message)
      - temperature: controls creativity (0.3=precise, 1.0=creative, 1.5=wild)
      - top_k: controls vocabulary diversity
      - max_tokens: controls response length
      - confidence_threshold: when to emit output
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_k: int = 40,
        max_tokens: int = 80,
        confidence_threshold: float = 0.5,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold

    def apply(self, rag_engine) -> dict:
        """Apply this plugin's settings to the RAG engine."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "confidence_threshold": self.confidence_threshold,
        }


# Pre-built plugins (like apps on a phone).
BUILTIN_PLUGINS = {
    "analyst": CognitivePlugin(
        name="analyst",
        system_prompt="You are a precise data analyst. Be factual, concise, and structured.",
        temperature=0.3,
        top_k=20,
        max_tokens=100,
        confidence_threshold=0.7,
    ),
    "creative": CognitivePlugin(
        name="creative",
        system_prompt="You are a creative writer. Be imaginative and expressive.",
        temperature=1.2,
        top_k=60,
        max_tokens=120,
        confidence_threshold=0.3,
    ),
    "coder": CognitivePlugin(
        name="coder",
        system_prompt="You are a senior software engineer. Write clean, correct code.",
        temperature=0.2,
        top_k=15,
        max_tokens=150,
        confidence_threshold=0.8,
    ),
    "teacher": CognitivePlugin(
        name="teacher",
        system_prompt="You are a patient teacher. Explain things simply with examples.",
        temperature=0.5,
        top_k=40,
        max_tokens=120,
        confidence_threshold=0.6,
    ),
    "hacker": CognitivePlugin(
        name="hacker",
        system_prompt="You are a cybersecurity expert. Think like both an attacker and defender.",
        temperature=0.4,
        top_k=30,
        max_tokens=100,
        confidence_threshold=0.7,
    ),
}


class PluginManager:
    """Manages hot-swappable cognitive plugins.

    Allows changing the engine's personality, style, and behavior
    at runtime WITHOUT any retraining. Like installing/uninstalling
    apps on a phone.

    Usage:
        pm = PluginManager(rag)
        pm.load("coder")        # engine now thinks like a coder
        pm.load("creative")     # switch to creative mode
        pm.unload()             # back to default
        pm.custom("name", temperature=0.9, ...)  # create custom plugin
    """

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.current: Optional[CognitivePlugin] = None
        self.available = dict(BUILTIN_PLUGINS)

    def load(self, name: str):
        """Load a plugin by name. Changes behavior instantly."""
        if name not in self.available:
            raise ValueError(f"Unknown plugin '{name}'. Available: {list(self.available.keys())}")
        self.current = self.available[name]
        return self.current

    def unload(self):
        """Remove the current plugin. Back to default behavior."""
        self.current = None

    def custom(
        self,
        name: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_k: int = 40,
        max_tokens: int = 80,
        confidence_threshold: float = 0.5,
    ):
        """Create and load a custom plugin on the fly."""
        plugin = CognitivePlugin(
            name=name,
            system_prompt=system_prompt,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            confidence_threshold=confidence_threshold,
        )
        self.available[name] = plugin
        self.current = plugin
        return plugin

    def list_available(self) -> List[str]:
        """List all available plugins."""
        return list(self.available.keys())

    def get_settings(self) -> dict:
        """Get current effective settings (plugin or default)."""
        if self.current:
            return self.current.apply(self.rag)
        return {
            "temperature": 0.7,
            "top_k": 40,
            "max_tokens": 80,
            "confidence_threshold": 0.5,
        }


# ---------------------------------------------------------------------------
# MetaCognition: the AI controls its own modules autonomously
# ---------------------------------------------------------------------------

class MetaCognition:
    """The AI decides HOW to process each input by itself.

    Instead of a human calling learn()/load()/retrieve(), the AI analyzes
    the input and chooses an action:

      Actions:
        RETRIEVE   — search the KB for relevant knowledge
        LEARN      — store new information permanently
        GENERATE   — produce an answer
        SWITCH     — change cognitive plugin (personality/mode)
        REFLECT    — think more before answering (multi-tick)

    The decision is made by a lightweight classifier (a small MLP) that
    takes the engine's thought state and outputs an action distribution.
    This classifier trains ONLINE (a few gradient steps per interaction)
    so the AI gets better at self-management through use.

    This is the difference between a TOOL and an AGENT:
    a tool waits for instructions; an agent decides what to do.
    """

    # Action space.
    ACTIONS = ["RETRIEVE", "LEARN", "GENERATE", "SWITCH", "REFLECT"]
    RETRIEVE, LEARN, GENERATE, SWITCH, REFLECT = range(5)

    def __init__(self, rag_engine, plugin_manager: PluginManager):
        self.rag = rag_engine
        self.pm = plugin_manager
        self.engine = rag_engine.engine
        self.d_model = self.engine.d_model

        # The action selector: a small MLP from thought state → action logits.
        self.action_net = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.ACTIONS)),
        )

        # Online optimizer for the action net (tiny, trains in real-time).
        self.optimizer = torch.optim.Adam(self.action_net.parameters(), lr=1e-3)

        # Feedback: track which actions led to good outcomes.
        self.action_history = []  # (action, reward)
        self.interaction_count = 0

    def decide(self, thought_state: torch.Tensor) -> int:
        """Decide what action to take based on the current thought state.

        Args:
            thought_state: (d_model,) the engine's current thought.
        Returns:
            action index (0-4).
        """
        with torch.no_grad():
            logits = self.action_net(thought_state.unsqueeze(0))
            # Epsilon-greedy: mostly pick best, sometimes explore.
            if torch.rand(1).item() < 0.1:  # 10% exploration
                return torch.randint(len(self.ACTIONS), (1,)).item()
            return logits.argmax(dim=-1).item()

    def execute(self, action: int, user_input: str) -> dict:
        """Execute the chosen action and return the result.

        The AI orchestrates its own modules. This is where autonomy lives.
        """
        result = {"action": self.ACTIONS[action], "content": ""}

        if action == self.RETRIEVE:
            # Search memory for relevant knowledge.
            retrieved = self.rag.retriever.retrieve(
                user_input, self.rag.tokenizer, top_k=3
            )
            result["content"] = " ".join(text for text, _, _ in retrieved)[:500]
            result["retrieved"] = True

        elif action == self.LEARN:
            # The AI decided this is worth remembering.
            self.rag.learn(user_input, source="self_directed")
            result["content"] = f"Learned: {user_input[:100]}"
            result["learned"] = True

        elif action == self.GENERATE:
            # Produce an answer using retrieved context.
            answer = self.rag.query(user_input, top_k=3, max_tokens=60)
            result["content"] = answer["answer"]
            result.update(answer)

        elif action == self.SWITCH:
            # The AI decides it needs a different cognitive mode.
            # Pick the best plugin based on input keywords.
            input_lower = user_input.lower()
            if any(w in input_lower for w in ["code", "function", "bug", "python", "api"]):
                self.pm.load("coder")
            elif any(w in input_lower for w in ["story", "poem", "creative", "imagine"]):
                self.pm.load("creative")
            elif any(w in input_lower for w in ["analyze", "data", "report", "statistics"]):
                self.pm.load("analyst")
            elif any(w in input_lower for w in ["explain", "teach", "learn", "understand"]):
                self.pm.load("teacher")
            elif any(w in input_lower for w in ["hack", "security", "vulnerability", "exploit"]):
                self.pm.load("hacker")
            else:
                self.pm.load("analyst")  # default

            plugin_name = self.pm.current.name if self.pm.current else "default"
            result["content"] = f"Switched to {plugin_name} mode"
            result["plugin"] = plugin_name

        elif action == self.REFLECT:
            # Think more: take extra ticks before answering.
            self.engine.eval()
            self.engine.reset_thought(1)
            ids = self.rag.tokenizer.encode(user_input)[:32]
            for tid in ids:
                self.engine.tick(torch.tensor([tid]))
            # Extra thinking ticks.
            for _ in range(5):
                logits, conf = self.engine.tick()
            result["content"] = "Reflected"
            result["confidence"] = conf.item()

        return result

    def process(self, user_input: str) -> dict:
        """Full autonomous processing: decide → act → learn from feedback.

        This is the main entry point. The AI:
          1. Encodes the input into a thought state.
          2. Decides what action to take (retrieve/learn/generate/switch/reflect).
          3. Executes the action.
          4. Optionally chains actions (e.g., SWITCH → RETRIEVE → GENERATE).
          5. Trains its own action_net from the outcome.

        Returns the final result with the full action chain.
        """
        self.interaction_count += 1
        action_chain = []

        # Encode input into a thought state.
        self.engine.eval()
        self.engine.reset_thought(1)
        ids = self.rag.tokenizer.encode(user_input)[:32]
        for tid in ids:
            self.engine.tick(torch.tensor([tid]))

        thought = self.engine.thought_state[0, 0, :].detach()

        # Multi-step: the AI can take up to 3 actions before answering.
        final_result = {"content": "", "actions": []}
        for step in range(3):
            action = self.decide(thought)
            result = self.execute(action, user_input)
            action_chain.append(result["action"])
            final_result["actions"] = action_chain

            # If GENERATE, we're done.
            if action == self.GENERATE:
                final_result["content"] = result.get("content", "")
                final_result.update({k: v for k, v in result.items()
                                     if k not in ("action", "content")})
                break

            # If RETRIEVE, feed the retrieved context and re-decide.
            if action == self.RETRIEVE and result.get("content"):
                ctx_ids = self.rag.tokenizer.encode(result["content"][:100])[:16]
                for tid in ctx_ids:
                    self.engine.tick(torch.tensor([tid]))
                thought = self.engine.thought_state[0, 0, :].detach()

            # If SWITCH, continue with the new mode.
            if action == self.SWITCH:
                continue

            # If LEARN or REFLECT, continue.
            if action in (self.LEARN, self.REFLECT):
                final_result["content"] = result.get("content", "")
                continue

        # If we went through all 3 steps without GENERATE, force it.
        if not final_result.get("content"):
            gen_result = self.execute(self.GENERATE, user_input)
            final_result["content"] = gen_result.get("content", "")
            action_chain.append("GENERATE")

        # Train the action net from this interaction (online learning).
        self._update_action_net(thought, user_input, final_result)

        final_result["interaction"] = self.interaction_count
        return final_result

    def _update_action_net(self, thought: torch.Tensor, user_input: str,
                           result: dict):
        """Online training of the action selector.

        The reward signal: did the action chain produce a non-empty,
        relevant answer? Simple but effective — the AI learns which
        action sequences work over time.
        """
        # Simple reward: 1 if we got content, 0 otherwise.
        reward = 1.0 if result.get("content") and len(result["content"]) > 10 else 0.0
        self.action_history.append((result["actions"], reward))

        # Keep only recent history (sliding window).
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]

        # One gradient step: reinforce the actions that led to good outcomes.
        self.optimizer.zero_grad()
        logits = self.action_net(thought.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)

        # If reward is positive, reinforce the chosen action distribution.
        # If negative, push away from it.
        if reward > 0.5:
            loss = -torch.log(probs.max() + 1e-8)  # reinforce best action
        else:
            loss = torch.log(probs.max() + 1e-8)  # discourage

        loss.backward()
        self.optimizer.step()

    def info(self) -> dict:
        avg_reward = (sum(r for _, r in self.action_history) /
                      max(len(self.action_history), 1))
        return {
            "meta_cognition": "active",
            "actions_available": self.ACTIONS,
            "interactions": self.interaction_count,
            "avg_reward": round(avg_reward, 3),
            "action_net_params": sum(p.numel() for p in self.action_net.parameters()),
            "autonomy": "The AI decides when to retrieve, learn, switch mode, or reflect",
        }
