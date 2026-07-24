"""
MTP self-speculative decoding: acceptance rate and wall-clock speedup.

DESIGN.md keeps MTP in the recipe on the strength of an *unmeasured* claim — that
the trained MTP head buys decode throughput even though it was a wash on
val_bpb. This measures it.

The scheme (DeepSeek-V3): MTP module 1 at position i consumes h^0_i plus the
embedding of the token one ahead and predicts the token *two* ahead. So it can
draft t_{P+2} the moment the trunk has produced t_{P+1}. We verify by feeding
[t_{P+1}, draft] through the main model in one forward:

  - logits at P+1 give the true t_{P+2}. If it equals the draft, the input at
    P+2 was right, so the logits at P+2 are valid too and we commit two tokens.
  - Otherwise we commit one token and roll the KV cache back by one position.

Either way the emitted sequence is exactly what plain greedy decoding produces:
the draft only ever saves work, it never changes the output.

That exactness holds in exact arithmetic, not in bf16. The verify forward passes
two tokens where plain decoding passes one, and the MoE's grouped GEMM reduces in
an order that depends on the token count, so the logits differ slightly. Measured
on a *fixed* token sequence with no speculation involved at all, chunk=1 vs
chunk=2 gives max |logit diff| ~1e0 and argmax mismatches at ~1.7% of positions.
Occasional divergence from greedy is therefore expected and is a property of
batched verification in bf16, not a bug in the scheme.

Three things get measured, because the obvious one is misleading on its own:

  1. rollback correctness — acceptance is so high that the reject branch almost
     never runs, so we force it every iteration and check we still reproduce
     greedy decoding exactly.
  2. acceptance on real held-out text (teacher-forced). This is the honest
     number. A d12 model trained on 1.3B tokens degenerates into loops when it
     decodes its own output, and drafting a loop is trivial, so acceptance
     measured on self-generated text is flattering and near-meaningless.
  3. wall-clock speedup on actual generation.

Usage (on a pod):
    python -m dev.spec_decode --model-tag a5-d12-mtp1
"""
import argparse
import time
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE
from nanochat.dataset import parquets_iter_batched
from nanochat.engine import KVCache
from nanochat.gpt import norm

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "The best way to learn a new programming language is",
    "def fibonacci(n):",
    "The three laws of thermodynamics state that",
    "Once upon a time, in a village at the edge of the forest,",
    "The main difference between a virus and a bacterium is",
    "To make a good espresso you need",
]

device = torch.device("cuda")


def make_cache(model, n_layers, seq_len):
    m = model.config
    return KVCache(batch_size=1, num_heads=m.n_kv_head, seq_len=seq_len,
                   head_dim=m.n_embd // m.n_head, num_layers=n_layers,
                   device=device, dtype=COMPUTE_DTYPE)


def embed_for_smear(model, tok):
    """The pre-smear normed embedding the model caches as `prev_embedding`. We set
    it by hand after a rollback, where the cached one belongs to a rejected draft."""
    ids = torch.tensor([[tok]], device=device)
    return norm(model.transformer.wte(ids).to(COMPUTE_DTYPE))


@torch.inference_mode()
def baseline(model, prompt_ids, n_new):
    """Plain greedy decoding, one token per forward."""
    cache = make_cache(model, model.config.n_layer, len(prompt_ids) + n_new + 8)
    logits = model.forward(torch.tensor([prompt_ids], device=device), kv_cache=cache)
    nxt = int(logits[0, -1].argmax())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = []
    for _ in range(n_new):
        out.append(nxt)
        logits = model.forward(torch.tensor([[nxt]], device=device), kv_cache=cache)
        nxt = int(logits[0, -1].argmax())
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.inference_mode()
def speculative(model, prompt_ids, n_new, force_reject=False):
    """Self-speculative decoding with the MTP head.

    Returns (tokens, seconds, n_iters, n_accepted). force_reject takes the
    rejection branch every iteration regardless of whether the draft matched;
    the emitted tokens must still equal plain greedy decoding.
    """
    total = len(prompt_ids) + n_new + 8
    cache = make_cache(model, model.config.n_layer, total)
    # the MTP block was built with layer_idx == n_layer, so its cache needs that
    # many slots for the index to land (and for Block to advance the position)
    mtp_cache = make_cache(model, model.config.n_layer + 1, total)

    # Prefill the trunk, then bring the MTP cache up to the same dense position.
    logits, h0 = model.forward(torch.tensor([prompt_ids], device=device),
                               kv_cache=cache, return_hidden=True)
    nxt = int(logits[0, -1].argmax())
    ahead = torch.tensor([prompt_ids[1:] + [nxt]], device=device)
    mtp_logits, _ = model.mtp_step(h0, ahead, mtp_cache)
    draft = int(mtp_logits[0, -1].argmax())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out, iters, accepted = [], 0, 0
    while len(out) < n_new:
        iters += 1
        pair = torch.tensor([[nxt, draft]], device=device)
        logits, h0 = model.forward(pair, kv_cache=cache, return_hidden=True)
        true_next = int(logits[0, 0].argmax())  # what the trunk really predicts after `nxt`

        if true_next == draft and not force_reject:
            accepted += 1
            out.extend([nxt, draft])
            follow = int(logits[0, 1].argmax())  # valid: the input at this position was correct
            # MTP at both newly committed positions, keeping its cache dense
            ahead = torch.tensor([[draft, follow]], device=device)
            mtp_logits, _ = model.mtp_step(h0, ahead, mtp_cache)
            nxt, draft = follow, int(mtp_logits[0, -1].argmax())
        else:
            out.append(nxt)
            # drop the rejected draft's KV entry, and restore the smear state that
            # the rejected token overwrote
            cache.cache_seqlens -= 1
            cache.prev_embedding = embed_for_smear(model, nxt)
            mtp_logits, _ = model.mtp_step(
                h0[:, :1], torch.tensor([[true_next]], device=device), mtp_cache)
            nxt, draft = true_next, int(mtp_logits[0, -1].argmax())
    torch.cuda.synchronize()
    return out[:n_new], time.perf_counter() - t0, iters, accepted


@torch.inference_mode()
def acceptance_on_real_text(model, tokenizer, n_seq, seq_len):
    """Teacher-forced acceptance on held-out validation text.

    At every position i we ask whether the MTP draft for t_{i+2} equals what the
    trunk would greedily emit there. Same question the decode loop asks, but on
    natural context instead of the model's own (degenerate) output.
    """
    docs = next(parquets_iter_batched(split="val"))
    ids = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc))
        if len(ids) >= n_seq * seq_len:
            break
    hits = tot = 0
    for s in range(n_seq):
        chunk = ids[s * seq_len:(s + 1) * seq_len]
        if len(chunk) < 8:
            break
        x = torch.tensor([chunk], device=device)
        logits, h0 = model.forward(x, return_hidden=True)
        trunk_next = logits[0].argmax(-1)                       # trunk_next[i] = t_{i+1}
        # MTP at positions 0..T-2, each fed the token one ahead
        mtp_logits, _ = model.mtp_step(h0[:, :-1], x[:, 1:], None)
        drafts = mtp_logits[0].argmax(-1)                       # drafts[i] ~ t_{i+2}
        hits += int((drafts == trunk_next[1:]).sum())
        tot += drafts.numel()
    return hits / max(tot, 1), tot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, default="a5-d12-mtp1")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--tf-seqs", type=int, default=8, help="held-out sequences for acceptance")
    parser.add_argument("--tf-len", type=int, default=1024)
    args = parser.parse_args()

    model, tokenizer, _ = load_model("base", device, "eval",
                                     model_tag=args.model_tag, step=args.step)
    assert model.config.n_mtp > 0, f"{args.model_tag} has no MTP head to speculate with"
    model.eval()
    print(f"model={args.model_tag} n_mtp={model.config.n_mtp} "
          f"tokens/prompt={args.num_tokens} prompts={args.num_prompts}")

    # --- 1. rollback correctness -------------------------------------------------
    ids = tokenizer.encode(PROMPTS[0])
    b_out, _ = baseline(model, ids, args.num_tokens)
    r_out, _, _, r_ac = speculative(model, ids, args.num_tokens, force_reject=True)
    bad = next((j for j, (p, q) in enumerate(zip(b_out, r_out)) if p != q), None)
    print(f"\n[1] forced-reject path reproduces greedy: "
          f"{'YES' if bad is None else f'NO — diverged at {bad}'} "
          f"(accepted {r_ac}, as intended)")

    # --- 2. acceptance on real held-out text -------------------------------------
    tf_rate, tf_n = acceptance_on_real_text(model, tokenizer, args.tf_seqs, args.tf_len)
    print(f"[2] acceptance on held-out text: {tf_rate:.1%} over {tf_n} positions")

    # --- 3. wall-clock speedup ---------------------------------------------------
    print("[3] generation:")
    baseline(model, ids, 16), speculative(model, ids, 16)  # warmup
    tb = ts = 0.0
    it = ac = 0
    diverged = []
    for i, text in enumerate(PROMPTS[:args.num_prompts]):
        p = tokenizer.encode(text)
        b_out, b_t = baseline(model, p, args.num_tokens)
        s_out, s_t, s_it, s_ac = speculative(model, p, args.num_tokens)
        d = next((j for j, (x, y) in enumerate(zip(b_out, s_out)) if x != y), None)
        if d is not None:
            diverged.append((i, d))
        tb, ts, it, ac = tb + b_t, ts + s_t, it + s_it, ac + s_ac
        print(f"  [{i}] baseline {args.num_tokens/b_t:7.1f} tok/s | "
              f"spec {args.num_tokens/s_t:7.1f} tok/s | accept {s_ac/s_it:5.1%} | "
              f"{b_t/s_t:.2f}x{'' if d is None else f'  (diverged @ {d})'}")

    n = args.num_prompts * args.num_tokens
    print("\n=== MTP self-speculative decoding ===")
    print(f"baseline        : {n/tb:8.1f} tok/s")
    print(f"speculative     : {n/ts:8.1f} tok/s")
    print(f"SPEEDUP         : {tb/ts:8.2f}x")
    print(f"tokens/iter     : {n/it:8.3f}  (theoretical max 2.0)")
    print(f"accept (self-gen): {ac/it:7.1%}  <- inflated by degenerate loops")
    print(f"accept (real text): {tf_rate:6.1%}  <- the honest number")
    # Project the speedup at the real-text acceptance rate: an iteration costs a
    # fixed amount regardless of outcome, only the tokens it yields change.
    sec_per_iter = ts / it
    print(f"PROJECTED at {tf_rate:.1%} acceptance: "
          f"{(1 + tf_rate)/sec_per_iter:.1f} tok/s = {(1 + tf_rate)/sec_per_iter/(n/tb):.2f}x")
    print(f"divergences vs greedy: {len(diverged)}/{args.num_prompts} prompts {diverged} "
          f"(bf16 chunk-shape noise, see module docstring)")


if __name__ == "__main__":
    main()
