#!/usr/bin/env python
"""Train the salience head by perturbation of trajectory.

The salience head learns INTRINSICALLY: at each tick with memory active, it
predicts how much the memory injection will perturb the thought state. The
signal is ||h_after - h_before||, normalized by a running max. No external
labels, no REINFORCE — just the dynamical system learning its own sensitivity.

This runs the slow per-tick path (memory is only in `tick`, not `tick_chunk`).
~25 tok/s on CPU. Budget: 50k tokens (~30 min).

Usage:
    python scripts/train_salience_head.py [--tokens 50000] [--budget-corpus 200000]
"""
import argparse, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import experiments.edt_ab.ablib as ablib
from fractus.memory import PersistentMemory

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")
SALIENCE_LAMBDA = 0.01  # weight of salience loss vs CE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50_000, help="training tokens")
    ap.add_argument("--budget-corpus", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(args.seed)

    print("=== Salience Head Training (perturbation of trajectory) ===", flush=True)
    split = ablib.load_corpus(CORPUS, n_train=args.budget_corpus,
                              n_holdout=10_000, n_phase1=int(args.budget_corpus * 0.15))
    tokens = split["train"][:args.tokens]
    holdout = split["holdout"]

    # Build engine + attach memory.
    eng = ablib.build_engine(seed=args.seed)
    mem = PersistentMemory(d_model=128, max_memories=128)
    eng.attach_memory(mem)
    eng.memory_active = True

    # Snapshot salience head weights to detect movement.
    sal_before = eng.salience_head.weight.detach().clone()
    sal_bias_before = eng.salience_head.bias.detach().clone()

    opt = torch.optim.AdamW(eng.parameters(), lr=args.lr, weight_decay=0.01)
    eng.train()
    eng.reset_thought(batch_size=1)

    t0 = time.time()
    total_ce, total_sal, n = 0.0, 0.0, 0
    for t in range(len(tokens) - 1):
        obs = tokens[t:t + 1]
        target = tokens[t + 1:t + 2]
        logits, conf = eng.tick(obs)
        ce = F.cross_entropy(logits, target)
        sal_loss = getattr(eng, 'last_salience_loss', torch.tensor(0.0))
        loss = ce + SALIENCE_LAMBDA * sal_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()

        total_ce += ce.item()
        total_sal += sal_loss.item()
        n += 1

        if (t + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / max(elapsed, 1)
            print(f"  t={t+1:>6} ce={total_ce/n:.3f} sal={total_sal/n:.4f} "
                  f"mem={len(mem)} pert_max={eng._pert_max:.3f} "
                  f"{rate:.0f} tok/s", flush=True)

    elapsed = time.time() - t0
    # Check if the salience head actually learned.
    sal_after = eng.salience_head.weight.detach()
    sal_bias_after = eng.salience_head.bias.detach()
    weight_delta = (sal_after - sal_before).norm().item()
    bias_delta = (sal_bias_after - sal_bias_before).norm().item()

    print(f"\n{'='*60}", flush=True)
    print(f"Training done: {n} tokens in {elapsed/60:.1f}min ({n/elapsed:.0f} tok/s)", flush=True)
    print(f"Final CE: {total_ce/n:.3f}  Final salience loss: {total_sal/n:.4f}", flush=True)
    print(f"Memories consolidated: {len(mem)}", flush=True)
    print(f"Salience head weight delta: {weight_delta:.6f} (should be > 0 if learned)", flush=True)
    print(f"Salience head bias delta: {bias_delta:.6f}", flush=True)
    print(f"Perturbation running max: {eng._pert_max:.4f}", flush=True)

    if weight_delta > 1e-6:
        print("VERDICT: salience head LEARNED (weights moved).", flush=True)
    else:
        print("VERDICT: salience head did NOT learn (weights unchanged).", flush=True)

    # Evaluate: does the salience head now predict perturbation?
    eng.eval()
    eng.reset_thought(batch_size=1)
    predicted, actual = [], []
    with torch.no_grad():
        for t in range(min(500, len(tokens) - 1)):
            obs = tokens[t:t + 1]
            logits, _ = eng.tick(obs)
            predicted.append(torch.sigmoid(eng.salience_head(
                eng.thought_state[:, 0, :])).item())
            actual.append(getattr(eng, '_last_perturbation', 0.0))
    if max(actual) > 0:
        corr_n = min(len(predicted), len(actual))
        pm, am = sum(predicted[:corr_n])/corr_n, sum(actual[:corr_n])/corr_n
        cov = sum((p-pm)*(a-am) for p,a in zip(predicted[:corr_n], actual[:corr_n]))
        vp = sum((p-pm)**2 for p in predicted[:corr_n])
        va = sum((a-am)**2 for a in actual[:corr_n])
        import math as m
        denom = m.sqrt(vp * va) if vp > 0 and va > 0 else 0
        corr = cov / denom if denom > 0 else 0
        print(f"Correlation(predicted_salience, actual_perturbation) = {corr:.3f}", flush=True)
    else:
        print("No perturbations measured during eval — memory may be empty.", flush=True)


if __name__ == "__main__":
    main()
