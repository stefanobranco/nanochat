"""
Fused linear + softcap + cross-entropy.

The naive path materializes the full logits tensor in fp32. At d12 with
device-batch-size 16 that is (32768 tokens x 32768 vocab) x 4B = 4.3 GiB, built
once for the trunk head and again for the MTP head, then walked several more
times by the softcap and by cross-entropy. It is the single largest tensor in the
step and the reason device-batch-size 32 will not fit with MTP enabled.

Chunking the token dimension and computing the gradient *during* the forward (the
Liger trick) keeps the matmul count identical to the naive path — one forward
matmul plus two for the gradients — while never holding more than one chunk of
logits. So this is a memory-traffic win, not a recompute trade. Plain gradient
checkpointing would also cut the memory but would add a third full projection.

Backends, in preference order:
  liger    liger_kernel's fused_linear_cross_entropy, if importable
  chunked  the implementation below, pure PyTorch
  naive    materialize logits and call F.cross_entropy (reference / fallback)

The softcap `c * tanh(z / c)` is nanochat's, and its derivative is folded into the
gradient here: d/dz [c*tanh(z/c)] = 1 - tanh^2(z/c) = 1 - (s/c)^2.
"""
import torch
import torch.nn.functional as F

_LIGER = None


def _try_liger():
    global _LIGER
    if _LIGER is None:
        try:
            from liger_kernel.ops.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyFunction,
            )
            _LIGER = LigerFusedLinearCrossEntropyFunction
        except Exception:
            _LIGER = False
    return _LIGER


class _ChunkedLinearCE(torch.autograd.Function):
    """Cross-entropy over a linear projection, chunked over tokens.

    Gradients are accumulated in the forward pass, so no chunk of logits has to
    survive into the backward. Only the (already small) input and weight
    gradients are carried.
    """

    @staticmethod
    def forward(ctx, h, weight, targets, softcap, ignore_index, chunk):
        N = h.size(0)
        valid = targets != ignore_index
        n_valid = valid.sum().clamp(min=1)
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        gh = torch.empty_like(h)
        gw = torch.zeros_like(weight, dtype=torch.float32)

        for i in range(0, N, chunk):
            hc = h[i:i + chunk]
            tc = targets[i:i + chunk]
            vc = valid[i:i + chunk]
            tgt = tc.clamp(min=0)

            z = (hc @ weight.t()).float()
            s = softcap * torch.tanh(z / softcap) if softcap else z

            lse = s.logsumexp(-1)
            picked = s.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            loss += ((lse - picked) * vc).sum()

            # d(loss)/d(s) = softmax(s) - onehot(target), zeroed on ignored rows
            g = (s - lse.unsqueeze(-1)).exp()
            g.scatter_add_(-1, tgt.unsqueeze(-1), -torch.ones_like(picked).unsqueeze(-1))
            g = g * vc.unsqueeze(-1)
            if softcap:
                g = g * (1.0 - (s / softcap) ** 2)  # through the tanh softcap
            g = g / n_valid  # mean reduction over valid targets

            gh[i:i + chunk] = (g.to(weight.dtype) @ weight)
            gw += g.t() @ hc.float()

        ctx.save_for_backward(gh, gw.to(weight.dtype))
        return loss / n_valid

    @staticmethod
    def backward(ctx, grad_out):
        gh, gw = ctx.saved_tensors
        return grad_out * gh, grad_out * gw, None, None, None, None


def linear_cross_entropy(h, weight, targets, softcap=15.0, ignore_index=-1,
                         backend="auto", chunk=8192):
    """Mean cross-entropy of `softcap_fn(h @ weight.T)` against `targets`.

    h: (N, C) — already normed. weight: (V, C). targets: (N,) with ignore_index.
    Equivalent to the naive path up to floating-point ordering.
    """
    if backend == "auto":
        backend = "liger" if _try_liger() else "chunked"

    if backend == "liger":
        fn = _try_liger()
        assert fn, "liger backend requested but liger_kernel is not importable"
        # Positional, against liger 0.8.1's signature:
        #   (_input, weight, target, bias, ce_weight, ignore_index,
        #    lse_square_scale, label_smoothing, reduction, softcap)
        # Function.apply takes no kwargs, so the order has to be exact.
        out = fn.apply(h, weight, targets, None, None, ignore_index,
                       0.0, 0.0, "mean", softcap if softcap else None)
        return out[0] if isinstance(out, tuple) else out

    if backend == "chunked":
        return _ChunkedLinearCE.apply(h, weight, targets, softcap, ignore_index, chunk)

    if backend == "naive":
        z = (h @ weight.t()).float()
        s = softcap * torch.tanh(z / softcap) if softcap else z
        return F.cross_entropy(s, targets, ignore_index=ignore_index, reduction="mean")

    raise ValueError(f"unknown backend {backend!r}")


def available_backend():
    return "liger" if _try_liger() else "chunked"
