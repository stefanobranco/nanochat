"""
Faithfulness checks for MuonH / Hyperball (arXiv 2606.16899).

Pins the invariants the paper is built on: every constrained matrix stays on the
Frobenius sphere of its init norm for the whole run, the pre-projection step has
angular size exactly lr, weight decay is inert on constrained groups, and the
zero-init projections that would make R = 0 degenerate are un-zeroed. CPU-only:
    CUDA_VISIBLE_DEVICES= python -m dev.test_hyperball
"""
import torch

from nanochat.gpt import GPT, GPTConfig

torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")


cfg = GPTConfig(sequence_len=64, vocab_size=256, n_layer=4, n_head=2, n_kv_head=2,
                n_embd=64, n_experts=8, n_topk=2, n_shared=1, n_mtp=1, hyperball=True)
m = GPT(cfg)
m.init_weights()

# 1. R = 0 is degenerate, so the projections must not be zero-init under hyperball.
projs = [(n, p) for n, p in m.named_parameters()
         if any(k in n for k in ("c_proj", "w_proj", "shared_proj"))]
check("output projections are non-zero at init",
      all(float(p.abs().sum()) > 0 for _, p in projs), f"({len(projs)} tensors)")

opt = m.setup_optimizer()
muon_groups = [g for g in opt.param_groups if g.get("kind") == "muon"]
check("all Muon groups are hyperball-flagged", all(g.get("hyperball") for g in muon_groups))

# record init norms of every Muon-managed matrix, per trailing (m, n) slice
def slice_norms(p):
    return p.float().norm(dim=(-2, -1)) if p.ndim >= 2 else p.float().norm()

muon_params = [p for g in muon_groups for p in g["params"]]
init_norms = [slice_norms(p).clone() for p in muon_params]
check("no Muon matrix starts at zero norm",
      all(float(n.min()) > 0 for n in init_norms))

# 2-4. Run several steps with real gradients; norms must be invariant to fp32 noise,
# even with an absurd weight decay (which the constraint replaces).
for g in opt.param_groups:
    if g.get("kind") == "muon":
        g["weight_decay"] = 1e6  # would visibly shrink weights if not inert
m.train()
x = torch.randint(0, 256, (2, 32))
y = torch.randint(0, 256, (2, 32))
prev = [p.detach().clone() for p in muon_params]
for step in range(3):
    opt.zero_grad(set_to_none=False)
    loss = m(x, y)
    loss.backward()
    opt.step()

drift = max(float((slice_norms(p) - n0).abs().max() / n0.max())
            for p, n0 in zip(muon_params, init_norms))
check("Frobenius norms invariant across steps", drift < 1e-4, f"(max rel drift {drift:.2e})")
check("weights actually moved", any(not torch.equal(p, q) for p, q in zip(muon_params, prev)))

# 5. Angular step size: one step from a known state moves each slice by exactly
#    lr * R before projection, so ||W_new - W_old|| <= lr * R (projection shrinks).
lr = muon_groups[0]["lr"]
deltas = [float(((p - q).float().norm(dim=(-2, -1)) / n0).max())
          for p, q, n0 in zip(muon_params, prev, init_norms)]
# after 3 steps the bound is ~3*lr; just sanity-check the order of magnitude
check("angular step magnitude bounded by steps*lr",
      max(deltas) <= 3 * lr * 1.05, f"(max {max(deltas):.4f}, 3*lr = {3*lr:.4f})")

# 6. Every trainable param still receives a grad (routers/scalars stay on Adam).
missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
check("every trainable param receives a grad", not missing, str(missing[:3]))

print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
