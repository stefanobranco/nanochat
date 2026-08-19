"""
WSM (Warmup-Stable-Merge, arXiv 2507.17634) evaluation: replace the LR-decay
phase with checkpoint merging over a trailing window of a CONSTANT-LR run.

The paper's finding is that decay schedules are emulatable as weighted averages
of stable-phase checkpoints, with the merge *duration* the key hyperparameter.
This evaluates the simple-average variant over several trailing windows of a run
trained with --warmdown-ratio=0 --save-every=N --save-model-only, and reports
val bpb per window next to the last raw checkpoint (no merge) as control.

    python -m dev.wsm_merge --model-tag e2-d12-wsm-stable --windows 2,4,6,8,10
"""
import argparse
import os
import re

import torch

from nanochat.checkpoint_manager import build_model
from nanochat.common import get_base_dir, COMPUTE_DTYPE
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_tokenizer, get_token_bytes

parser = argparse.ArgumentParser()
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--windows", type=str, default="1,2,4,6,8,10", help="trailing checkpoint counts to merge (1 = no merge)")
parser.add_argument("--device-batch-size", type=int, default=32)
parser.add_argument("--eval-tokens", type=int, default=80 * 524288)
args = parser.parse_args()

device = torch.device("cuda")
ckpt_dir = os.path.join(get_base_dir(), "base_checkpoints", args.model_tag)
steps = sorted(int(m.group(1)) for f in os.listdir(ckpt_dir)
               if (m := re.match(r"model_(\d+)\.pt$", f)))
print(f"{len(steps)} checkpoints: {steps}")

# build the model skeleton from the final checkpoint (also loads tokenizer/meta)
model, tokenizer, meta = build_model(ckpt_dir, steps[-1], device, phase="eval")
token_bytes = get_token_bytes(device=device)
seq_len = meta["max_seq_len"]
eval_steps = args.eval_tokens // (args.device_batch_size * seq_len)


def eval_state(state):
    model.load_state_dict(state, strict=True)
    model.eval()
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, seq_len, split="val", device=device)
    with torch.no_grad():
        return evaluate_bpb(model, loader, eval_steps, token_bytes)


results = {}
windows = sorted({int(w) for w in args.windows.split(",")})
for w in windows:
    take = steps[-w:]
    if len(take) < w:
        print(f"window {w}: not enough checkpoints, skipping")
        continue
    # running average in fp32 to avoid drift over many summands
    avg = None
    for s in take:
        sd = torch.load(os.path.join(ckpt_dir, f"model_{s:06d}.pt"), map_location=device)
        if avg is None:
            avg = {k: v.float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg[k] += v.float()
        del sd
    state = {k: (v / len(take)).to(dtype) for (k, v), dtype in
             zip(avg.items(), [model.state_dict()[k].dtype for k in avg])}
    del avg
    bpb = eval_state(state)
    results[w] = bpb
    print(f"merge window {w:2d} (steps {take[0]}..{take[-1]}): val_bpb {bpb:.6f}")

print("\n=== WSM summary ===")
for w, bpb in sorted(results.items()):
    print(f"window {w:2d}: {bpb:.6f}")
best = min(results, key=results.get)
print(f"best: window {best} at {results[best]:.6f}")
