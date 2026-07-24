"""
Faithfulness checks for AttnRes (Kimi, arXiv 2603.15031), Full variant.

These pin the properties the paper is specific about — the ones a plausible-looking
reimplementation gets wrong silently. Runs on CPU:
    CUDA_VISIBLE_DEVICES= python -m dev.test_attnres
"""
import torch

from nanochat.gpt import AttnRes, GPT, GPTConfig, norm

torch.manual_seed(0)
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}")


cfg = GPTConfig(sequence_len=64, vocab_size=256, n_layer=4, n_head=2, n_kv_head=2,
                n_embd=64, n_experts=8, n_topk=2, n_shared=1, attn_res=True)

print("AttnRes layer:")
layer = AttnRes(cfg)
srcs = [torch.randn(2, 8, 64) for _ in range(5)]

# 1. Zero-init => uniform attention => plain average of the sources. The paper
#    calls the zero init crucial precisely because it gives this starting point.
check("zero-init pseudo-query", float(layer.q.abs().sum()) == 0)
check("at init, output == mean of sources",
      torch.allclose(layer(srcs), torch.stack(srcs).mean(0), atol=1e-6))

# 2. Weights are a softmax over the DEPTH axis: convex combination, sums to 1.
#    With a trained (non-zero) query the output must stay in the sources' hull.
with torch.no_grad():
    layer.q.normal_(0, 1.0)
out = layer(srcs)
q = layer.q
logits = torch.stack([(norm(v) * q).sum(-1) for v in srcs], 0)
w = logits.softmax(0)
check("attention weights sum to 1 over depth", torch.allclose(w.sum(0), torch.ones(2, 8), atol=1e-6))
check("output == sum_i w_i * v_i", torch.allclose(out, (w.unsqueeze(-1) * torch.stack(srcs)).sum(0), atol=1e-5))

# 3. RMSNorm is on the KEY path only: values are summed un-normalized. Scaling one
#    source's magnitude must change the output (it would not if values were
#    normalized), while leaving the attention weights untouched.
big = [s.clone() for s in srcs]
big[2] = big[2] * 50.0
out_big = layer(big)
logits_big = torch.stack([(norm(v) * q).sum(-1) for v in big], 0)
check("scaling a source changes the output (values un-normalized)",
      not torch.allclose(out, out_big, atol=1e-3))
check("scaling a source leaves its attention weight unchanged (keys normalized)",
      torch.allclose(logits, logits_big, atol=1e-4))

# 4. One instance per sublayer, plus the final aggregation.
print("AttnRes model wiring:")
m = GPT(cfg)
m.init_weights()
n_q = sum(1 for n, _ in m.named_parameters() if "attn_res" in n)
check(f"2*n_layer+1 == {2 * cfg.n_layer + 1} pseudo-queries", n_q == 2 * cfg.n_layer + 1)

# 5. The resid/x0 lambdas must be frozen, not merely unused: the fused AdamW step
#    cannot handle grad=None, which is how the mHC run first crashed.
m.setup_optimizer()
check("resid/x0 lambdas frozen", not m.resid_lambdas.requires_grad and not m.x0_lambdas.requires_grad)
m.train()
# nanochat zero-inits every sublayer output projection, so at step 0 all sources
# except the embedding are identically zero AND the gradient back through them is
# zero — no pseudo-query would see signal. That is a property of the init instant,
# not of the mechanism, so nudge the projections off zero (as one optimizer step
# would) before asking whether gradients flow.
with torch.no_grad():
    for n, p in m.named_parameters():
        if "c_proj" in n or "w_proj" in n:
            p.normal_(0, 0.02)
loss = m(torch.randint(0, 256, (2, 32)), torch.randint(0, 256, (2, 32)))
loss.backward()
check("every trainable param receives a grad",
      not [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None])

# The first sublayer has exactly one source (the embedding), so its softmax is 1
# regardless of the query: that pseudo-query is mathematically dead. It must still
# get a real zero grad rather than None — short-circuiting the S==1 case would give
# it grad=None and crash the fused AdamW step, which is how the mHC run first died.
qgrads = [(n, float(p.grad.abs().sum())) for n, p in m.named_parameters() if "attn_res" in n]
check("first pseudo-query has grad 0 (single source, softmax is constant)",
      qgrads[0][1] == 0.0)
check("every other pseudo-query receives a nonzero grad",
      all(g > 0 for _, g in qgrads[1:]))

print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
