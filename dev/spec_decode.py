"""
MTP self-speculative decoding: acceptance rate and wall-clock speedup.

DESIGN.md keeps MTP in the recipe on the strength of an *unmeasured* claim — that
the trained MTP head buys faster decoding even though it was a quality wash on
val_bpb. This measures it.

The scheme (DeepSeek-V3): MTP module 1 at position i consumes h^0_i plus the
embedding of the token one ahead and predicts the token *two* ahead. So it can
draft t_{P+2} the moment the trunk has produced t_{P+1}. We then verify by
feeding [t_{P+1}, draft] through the main model in a single forward:

  - logits at P+1 give the true t_{P+2}. If it equals the draft, the input at
    P+2 was right, so the logits at P+2 are valid too and we commit two tokens.
  - Otherwise we commit one token and roll the KV cache back by one position.

Either way the emitted sequence is exactly what plain greedy decoding produces —
the draft only ever saves work, it never changes the output. The script asserts
that, so a speedup number can't come from silently decoding something else.

The MTP module keeps its own KV cache, held *dense* over token positions: during
training it attended to every position, so letting gaps appear at inference would
degrade the drafts and understate acceptance.

Usage (on a pod):
    python -m dev.spec_decode --model-tag a5-d12-mtp1
"""
import argparse
import time
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import norm

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, default="a5-d12-mtp1")
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--num-tokens", type=int, default=256, help="tokens to generate per prompt")
parser.add_argument("--num-prompts", type=int, default=8)
args = parser.parse_args()

device = torch.device("cuda")
model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=args.step)
assert model.config.n_mtp > 0, f"{args.model_tag} has no MTP head; nothing to speculate with"
model.eval()

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


def make_cache(n_layers, seq_len):
    m = model.config
    return KVCache(batch_size=1, num_heads=m.n_kv_head, seq_len=seq_len,
                   head_dim=m.n_embd // m.n_head, num_layers=n_layers,
                   device=device, dtype=COMPUTE_DTYPE)


def embed_for_smear(tok):
    """The pre-smear normed embedding the model caches as `prev_embedding`. We set
    it by hand after a rollback, where the cached one belongs to a rejected draft."""
    ids = torch.tensor([[tok]], device=device)
    return norm(model.transformer.wte(ids).to(COMPUTE_DTYPE))


@torch.inference_mode()
def baseline(prompt_ids, n_new):
    """Plain greedy decoding, one token per forward."""
    cache = make_cache(model.config.n_layer, len(prompt_ids) + n_new + 8)
    ids = torch.tensor([prompt_ids], device=device)
    logits = model.forward(ids, kv_cache=cache)
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
def speculative(prompt_ids, n_new, force_reject=False):
    """Self-speculative decoding with the MTP head. Returns (tokens, seconds,
    n_iters, n_accepted).

    force_reject takes the rejection branch on every iteration regardless of
    whether the draft matched. Acceptance is normally so high that the rollback
    path is barely exercised, so this is how we actually test it: the emitted
    tokens must still equal plain greedy decoding.
    """
    L = len(prompt_ids)
    total = L + n_new + 8
    cache = make_cache(model.config.n_layer, total)
    # the MTP block was built with layer_idx == n_layer, so its cache needs that
    # many slots for the index to land (and for Block to advance the position)
    mtp_cache = make_cache(model.config.n_layer + 1, total)

    # Prefill the trunk, then bring the MTP cache up to the same dense position.
    ids = torch.tensor([prompt_ids], device=device)
    logits, h0 = model.forward(ids, kv_cache=cache, return_hidden=True)
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
            h_pair = h0
            ahead = torch.tensor([[draft, follow]], device=device)
            mtp_logits, _ = model.mtp_step(h_pair, ahead, mtp_cache)
            nxt, draft = follow, int(mtp_logits[0, -1].argmax())
        else:
            out.append(nxt)
            # drop the rejected draft's KV entry, and restore the smear state that
            # the rejected token overwrote
            cache.cache_seqlens -= 1
            cache.prev_embedding = embed_for_smear(nxt)
            mtp_logits, _ = model.mtp_step(
                h0[:, :1], torch.tensor([[true_next]], device=device), mtp_cache)
            nxt, draft = true_next, int(mtp_logits[0, -1].argmax())
    torch.cuda.synchronize()
    return out[:n_new], time.perf_counter() - t0, iters, accepted


print(f"model={args.model_tag} n_mtp={model.config.n_mtp} "
      f"tokens/prompt={args.num_tokens} prompts={args.num_prompts}")

# warmup (kernel autotuning, lazy inits) so the first prompt isn't penalized
_ = baseline(tokenizer.encode(PROMPTS[0]), 16)
_ = speculative(tokenizer.encode(PROMPTS[0]), 16)

tb = ts = 0.0
it = ac = 0
for i, text in enumerate(PROMPTS[:args.num_prompts]):
    ids = tokenizer.encode(text)
    b_out, b_t = baseline(ids, args.num_tokens)
    s_out, s_t, s_it, s_ac = speculative(ids, args.num_tokens)
    assert b_out == s_out, (
        f"prompt {i}: speculative output diverged from greedy at position "
        f"{next(j for j, (x, y) in enumerate(zip(b_out, s_out)) if x != y)}")
    tb += b_t
    ts += s_t
    it += s_it
    ac += s_ac
    print(f"  [{i}] baseline {args.num_tokens/b_t:7.1f} tok/s | "
          f"spec {args.num_tokens/s_t:7.1f} tok/s | "
          f"accept {s_ac/s_it:5.1%} | speedup {b_t/s_t:.2f}x")

n = args.num_prompts * args.num_tokens
print("\n=== MTP self-speculative decoding ===")
print(f"outputs identical to greedy: YES ({n} tokens checked)")
print(f"baseline    : {n/tb:8.1f} tok/s")
print(f"speculative : {n/ts:8.1f} tok/s")
print(f"acceptance  : {ac/it:8.1%}  ({ac}/{it} drafts accepted)")
print(f"tokens/iter : {n/it:8.3f}  (theoretical max 2.0)")
print(f"SPEEDUP     : {tb/ts:8.2f}x")
