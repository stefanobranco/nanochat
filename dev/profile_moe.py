"""
MoE throughput profiler for SLAI. Answers: where does the MoE MFU gap go?
(dense d12 ~42% MFU vs MoE d12 ~14.5%). Measurement-first: confirm the
bottlenecks before optimizing, don't guess.

Run on a GPU pod:
    python -m dev.profile_moe                    # MoE d12, eager, full breakdown
    python -m dev.profile_moe --dense            # dense baseline for reference
    python -m dev.profile_moe --compile          # compiled wall-clock (matches training)

Outputs: (1) wall-clock throughput + MFU, (2) torch.profiler top ops by CUDA
time, (3) targeted micro-benchmarks for the three suspects (weight cast,
grouped GEMM, routing ops).
"""
import argparse
import torch
from nanochat.gpt import GPT, GPTConfig

H100_BF16_PEAK = 989e12  # SXM bf16 dense tensor-core peak

def build(depth, n_experts, device):
    head_dim = 128
    model_dim = ((depth * 64 + head_dim - 1) // head_dim) * head_dim
    cfg = GPTConfig(
        sequence_len=2048, vocab_size=32768, n_layer=depth,
        n_head=model_dim // head_dim, n_kv_head=model_dim // head_dim, n_embd=model_dim,
        n_experts=n_experts, n_topk=4, n_shared=1,
    )
    with torch.device(device):
        model = GPT(cfg)
    model.init_weights()
    model.train()
    return model

def throughput(model, B, T, device, steps=10, warmup=3, compile=False):
    fwd = torch.compile(model) if compile else model
    idx = torch.randint(0, 32768, (B, T), device=device)
    tgt = torch.randint(0, 32768, (B, T), device=device)
    for _ in range(warmup):
        fwd(idx, tgt).backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        fwd(idx, tgt).backward()
    end.record(); torch.cuda.synchronize()
    dt = start.elapsed_time(end) / steps / 1000.0
    tok_s = B * T / dt
    mfu = model.estimate_flops() * tok_s / H100_BF16_PEAK
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  step {dt*1000:.1f} ms | {tok_s:,.0f} tok/s | MFU {mfu*100:.1f}% | peak {peak_mem:.1f} GB")

def profile_ops(model, B, T, device):
    idx = torch.randint(0, 32768, (B, T), device=device)
    tgt = torch.randint(0, 32768, (B, T), device=device)
    for _ in range(3):
        model(idx, tgt).backward()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            model.zero_grad(set_to_none=True)
            model(idx, tgt).backward()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

def microbench(model, B, T, device):
    """Isolate the three suspects. Times are per-call, averaged."""
    layer = next(b.mlp for b in model.transformer.h if hasattr(b.mlp, "w_fc"))
    C = model.config.n_embd
    x = torch.randn(B * T, C, device=device, dtype=torch.bfloat16)
    def timed(fn, n=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / n
    # suspect 1: fp32->bf16 cast of all expert weights (happens every forward, x16 microsteps)
    t_cast = timed(lambda: (layer.w_fc.to(torch.bfloat16), layer.w_proj.to(torch.bfloat16)))
    # suspect 2: full MoE layer forward (grouped GEMM + routing together)
    t_layer = timed(lambda: layer(x.view(B, T, C)))
    print(f"  weight cast (both matrices): {t_cast:.3f} ms/call  x16 microsteps = {t_cast*16:.2f} ms/step")
    print(f"  full MoE layer forward:      {t_layer:.3f} ms/call")
    print(f"  => cast is ~{100*t_cast/t_layer:.0f}% of one layer forward")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--dense", action="store_true", help="n_experts=0 baseline")
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    n_experts = 0 if args.dense else 32
    B, T = args.device_batch_size, 2048
    model = build(args.depth, n_experts, device)
    tag = "dense" if args.dense else "MoE E32k4"
    print(f"=== {tag} d{args.depth} bs{B} {'(compiled)' if args.compile else '(eager)'} ===")
    print("[throughput]")
    throughput(model, B, T, device, compile=args.compile)
    if not args.compile:
        print("[top ops by CUDA time]")
        profile_ops(model, B, T, device)
        if not args.dense:
            print("[micro-benchmarks]")
            microbench(model, B, T, device)
