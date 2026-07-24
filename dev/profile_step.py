"""
Where does a MoE+MTP training step actually go?

Attribution by ablation rather than by profiler label: time the real step, then
time variants with one piece removed, and read the cost off the differences.
Profiler self-time is easy to misread across fused/compiled regions, whereas
"delete it and see" is unambiguous.

Variants:
  full        the real thing (MoE trunk + MTP head)
  no_mtp      n_mtp=0 -> isolates the whole MTP branch
  cheap_head  lm_head projects to 512 instead of the full vocab -> isolates
              lm_head matmul + softcap + cross-entropy (both heads)
  dense       MoE replaced by dense MLP -> isolates MoE dispatch overhead

Also prints the top CUDA ops for the full variant, and peak memory for each
(peak memory is what decides whether device-batch-size 32 fits).

Usage (on a pod, needs an idle GPU):
    python -m dev.profile_step --device-batch-size 16
"""
import argparse
import time

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig, Linear

parser = argparse.ArgumentParser()
parser.add_argument("--device-batch-size", type=int, default=16)
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--iters", type=int, default=6)
parser.add_argument("--compile", action="store_true", default=True)
parser.add_argument("--no-compile", dest="compile", action="store_false")
parser.add_argument("--variants", type=str, default="full,no_mtp,cheap_head,dense")
args = parser.parse_args()

device = torch.device("cuda")
VOCAB = 32768


def build(variant):
    d = args.depth
    cfg = dict(sequence_len=args.seq_len, vocab_size=VOCAB, n_layer=d,
               n_head=d // 2, n_kv_head=d // 2, n_embd=d * 64,
               n_experts=32, n_topk=4, n_shared=1, n_mtp=1)
    if variant == "no_mtp":
        cfg["n_mtp"] = 0
    if variant == "dense":
        cfg["n_experts"] = 0
    model = GPT(GPTConfig(**cfg)).to(device)
    model.init_weights()
    if variant == "cheap_head":
        # shrink the vocab projection but keep everything else identical; targets
        # are remapped into range by the caller
        # nanochat's Linear, not torch's: it casts the fp32 master weight to the
        # activation dtype, which a plain nn.Linear would not do
        model.lm_head = Linear(cfg["n_embd"], 512, bias=False).to(device)
        model.config.vocab_size = 512
    model.train()
    return model


def run(variant):
    torch.manual_seed(0)
    model = build(variant)
    compiled = torch.compile(model) if args.compile else model
    opt = model.setup_optimizer()
    V = 512 if variant == "cheap_head" else VOCAB
    B, T = args.device_batch_size, args.seq_len
    x = torch.randint(0, V, (B, T), device=device)
    y = torch.randint(0, V, (B, T), device=device)
    torch.cuda.reset_peak_memory_stats()

    def step():
        loss = compiled(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)  # keep grads allocated: Muon needs them

    for _ in range(3):  # warmup + compile
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    peak = torch.cuda.max_memory_allocated() / 2**30

    prof_rows = None
    if variant == "full":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
            step()
            torch.cuda.synchronize()
        prof_rows = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=14)

    del model, compiled, opt
    torch.cuda.empty_cache()
    return dt, peak, prof_rows


results = {}
for variant in args.variants.split(","):
    dt, peak, rows = run(variant)
    tok = args.device_batch_size * args.seq_len / dt
    results[variant] = (dt, peak, tok)
    print(f"{variant:11s} {dt*1000:8.1f} ms/microstep  {tok:9,.0f} tok/s  peak {peak:5.1f} GiB")
    if rows:
        print(rows)

if "full" in results:
    full_dt = results["full"][0]
    print("\n=== attribution (share of the full MoE+MTP microstep) ===")
    labels = {
        "no_mtp": "MTP branch (extra block + 2nd head + its CE)",
        "cheap_head": "lm_head matmul + softcap + CE, both heads",
        "dense": "MoE dispatch overhead vs dense MLP",
    }
    for variant, label in labels.items():
        if variant in results:
            delta = full_dt - results[variant][0]
            print(f"  {label:48s} {delta*1000:7.1f} ms  {delta/full_dt:6.1%}")
    print("\npeak memory (device-batch-size 32 needs to fit in ~79 GiB):")
    for variant, (_, peak, _) in results.items():
        print(f"  {variant:11s} {peak:5.1f} GiB")
