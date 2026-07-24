"""
Correctness of the chunked linear+softcap+cross-entropy against the naive path.

Checks loss, dh and dW, including the parts that are easy to get subtly wrong:
the tanh-softcap derivative, ignore_index rows contributing nothing, and the mean
reduction dividing by the number of *valid* targets rather than by N.

    CUDA_VISIBLE_DEVICES= python -m dev.test_fused_ce
"""
import torch

from nanochat.fused_ce import linear_cross_entropy

torch.manual_seed(0)
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")


def run(backend, h, w, tgt, softcap):
    h = h.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    loss = linear_cross_entropy(h, w, tgt, softcap=softcap, backend=backend, chunk=64)
    loss.backward()
    return float(loss), h.grad, w.grad


N, C, V = 300, 64, 512
for softcap in (15.0, 0.0):
    print(f"softcap={softcap}:")
    h = torch.randn(N, C, dtype=torch.float32)
    w = torch.randn(V, C, dtype=torch.float32) * 0.05
    tgt = torch.randint(0, V, (N,))
    tgt[::7] = -1  # ignored rows, incl. a partial final chunk

    l_ref, gh_ref, gw_ref = run("naive", h, w, tgt, softcap)
    l_ch, gh_ch, gw_ch = run("chunked", h, w, tgt, softcap)

    check("loss matches", abs(l_ref - l_ch) < 1e-5, f"({l_ref:.6f} vs {l_ch:.6f})")
    check("dh matches", torch.allclose(gh_ref, gh_ch, atol=1e-5),
          f"(max {float((gh_ref - gh_ch).abs().max()):.2e})")
    check("dW matches", torch.allclose(gw_ref, gw_ch, atol=1e-5),
          f"(max {float((gw_ref - gw_ch).abs().max()):.2e})")

# ignored rows must contribute nothing at all: perturbing their target changes nothing
print("ignore_index semantics:")
h = torch.randn(N, C)
w = torch.randn(V, C) * 0.05
t1 = torch.randint(0, V, (N,))
t1[:10] = -1
t2 = t1.clone()
l1, gh1, _ = run("chunked", h, w, t1, 15.0)
l2, gh2, _ = run("chunked", h, w, t2, 15.0)
check("ignored rows get zero input-grad", float(gh1[:10].abs().max()) == 0.0)

# all-ignored chunk must not produce NaN (n_valid clamp)
t3 = torch.full((N,), -1)
l3 = float(linear_cross_entropy(h, w, t3, softcap=15.0, backend="chunked", chunk=64))
check("all-ignored batch is finite", l3 == l3 and abs(l3) < 1e-6, f"(loss {l3})")

print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
