"""Decode-time surgery for Fractus CTE.

Breaks single-token and short-cycle attractors without changing weights.
Techniques: phase noise, thought noise, recent-token bans, frequency penalty,
cycle detection with hard scramble, periodic forced escape tokens.
"""
import math
import random
import torch

def generate_with_surgery(engine, tokenizer, prompt, max_new=48, temperature=1.15,
                          phase_noise=0.55, thought_noise=0.08, ban_window=20,
                          escape_every=4, freq_penalty=1.2):
    engine.eval()
    for blk in engine.blocks:
        if hasattr(blk, 'moe'):
            blk.moe.temperature = max(getattr(blk.moe, 'temperature', 1.0), 3.0)
    engine.reset_thought(1)
    ids = tokenizer.encode(prompt)[:64]
    with torch.no_grad():
        for t in ids:
            engine.tick(torch.tensor([t]))
        out = []
        cur = ids[-1] if ids else 0
        freq = {}
        for step in range(max_new):
            for blk in engine.blocks:
                if hasattr(blk, 'kuramoto_phases'):
                    blk.kuramoto_phases = torch.remainder(
                        blk.kuramoto_phases + phase_noise * torch.randn_like(blk.kuramoto_phases),
                        2 * math.pi)
            engine.thought_state = 0.9 * engine.thought_state + thought_noise * torch.randn_like(engine.thought_state)
            if step > 0 and escape_every and step % escape_every == 0:
                banned = set(out[-ban_window:])
                esc = random.randint(0, 50256)
                for _ in range(40):
                    esc = random.randint(0, 50256)
                    if esc not in banned and esc != 50256:
                        break
                engine.tick(torch.tensor([esc]))
                out.append(esc)
                freq[esc] = freq.get(esc, 0) + 1
                cur = esc
                for blk in engine.blocks:
                    if hasattr(blk, 'kuramoto_phases'):
                        blk.kuramoto_phases = torch.rand_like(blk.kuramoto_phases) * 2 * math.pi
                continue
            logits, _ = engine.tick(torch.tensor([cur]))
            l = logits[0].float().clone()
            for prev in out[-ban_window:]:
                l[prev] = -1e9
            for tid, c in freq.items():
                l[tid] -= freq_penalty * c
            topv, topi = torch.topk(l / max(temperature, 1e-5), 150)
            mask = torch.isfinite(topv)
            topv, topi = topv[mask], topi[mask]
            if topv.numel() == 0:
                nxt = int(torch.argmax(logits[0]).item())
            else:
                nxt = int(topi[torch.multinomial(torch.softmax(topv, -1), 1)].item())
            out.append(nxt)
            freq[nxt] = freq.get(nxt, 0) + 1
            cur = nxt
    return tokenizer.decode(out), out
