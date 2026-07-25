"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.fused_ce import linear_cross_entropy

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # MoE (DeepSeek-V3 recipe). n_experts=0 means dense, identical to baseline.
    n_experts: int = 0
    n_topk: int = 4 # active routed experts per token
    n_shared: int = 1 # always-on shared experts
    expert_hidden: int = 0 # hidden dim per expert; 0 = n_embd (i.e. 1/4 of the dense 4x MLP)
    moe_first_dense: int = 1 # keep this many initial layers dense
    router_bias_update_rate: float = 1e-3 # aux-loss-free balancing bias step size
    router_affinity: str = "sigmoid" # sigmoid (DSv3) | sqrtsoftplus (DSv4)
    # mHC (DSv4 manifold-constrained hyper-connections). 1 = off (plain residual).
    n_streams: int = 1
    mhc_sinkhorn_iters: int = 5
    # AttnRes (Kimi, arXiv 2603.15031). Full variant: softmax attention over every
    # earlier sublayer output, replacing the residual stream. Competes with mHC.
    attn_res: bool = False
    # MTP (DeepSeek-V3 multi-token prediction). 0 = off. Sequential modules, each
    # a transformer block + projection, sharing wte + lm_head. Training-only aux loss.
    n_mtp: int = 0
    mtp_weight: float = 0.3
    # Fuse the vocab projection into the loss instead of materializing fp32 logits.
    # MTP makes this matter twice over. Training only (the 'mean' reduction).
    fused_ce: bool = False


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


# NOTE: we used to wrap torch._grouped_mm in a custom autograd.Function because
# autograd support was missing in torch 2.9. As of 2.12 the public F.grouped_mm
# carries its own backward, verified bit-identical to the manual dX/dW formulas we
# had written. Dropping the Function also drops two graph breaks per MoE layer,
# which lets inductor fuse the surrounding cast/relu-square work.


class _GatherPermute(torch.autograd.Function):
    """Materialize the expert-sorted, padded activation buffer with a SINGLE gather.

    The obvious spelling — zeros(P,C).index_copy_(0, pos, xf[src]) — moves P*C
    activations three times (memset, gather into sorted order, scatter into padded
    order). Instead we scatter the *indices* (P int64s, ~1MB) once and let one
    index_select do the data movement: 1 read + 1 write.

    The catch is autograd. index_select's own backward would index_add over every
    row of gidx, including pad rows and the unused tail — and grouped_mm leaves the
    tail of its dx output uninitialized, so that garbage would land in xf. We supply
    the backward ourselves, restricted to the real (pos -> src) pairs, which is
    exactly what index_copy's backward did and costs the same.

    That hazard is measured, not theoretical: running F.grouped_mm with offsets that
    stop short of the tensor and inspecting dx beyond offs[-1] gives all-zeros on a
    fresh buffer but |dx| up to ~2e1 on reused allocations. A plain index_select
    here would corrupt gradients only intermittently, under memory pressure.
    """
    @staticmethod
    def forward(ctx, xf, gidx, perm, pad_idx, K):
        ctx.save_for_backward(perm)
        ctx.n_rows, ctx.K = xf.size(0), K
        xs = xf.index_select(0, gidx)
        xs.index_fill_(0, pad_idx, 0) # pad rows enter the GEMM; they must be exactly zero
        return xs

    @staticmethod
    def backward(ctx, g):
        # perm[n*K + k] is the padded slot holding token n's k-th expert, so gathering
        # through it lands a token's K contributions adjacent and the accumulation is a
        # plain reduction over K — no atomics, and pad/tail rows are never touched.
        (perm,) = ctx.saved_tensors
        return g.index_select(0, perm).view(ctx.n_rows, ctx.K, -1).sum(1), None, None, None, None


class MoEMLP(nn.Module):
    """
    DeepSeek-V3-style MoE FFN: fine-grained routed experts + shared expert(s).
    Sigmoid routing affinities; top-k selection uses affinity + per-expert bias,
    but gate weights use the raw affinities (bias steers load only). The bias is
    updated online toward uniform expert load (aux-loss-free balancing, no
    balance loss term).
    Experts live in two stacked 3D tensors and dispatch through a single grouped
    GEMM per projection (sorted token order, on-GPU offsets, zero CPU syncs).
    Muon handles the 3D params fine: its fused step is batched over leading dims.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_experts > 0 and config.n_topk <= config.n_experts
        H = config.expert_hidden if config.expert_hidden > 0 else config.n_embd
        self.n_experts = config.n_experts
        self.n_topk = config.n_topk
        self.bias_update_rate = config.router_bias_update_rate
        assert config.router_affinity in ("sigmoid", "sqrtsoftplus")
        self.router_affinity = config.router_affinity
        self.router = Linear(config.n_embd, config.n_experts, bias=False)
        self.w_fc = nn.Parameter(torch.empty(config.n_experts, config.n_embd, H))
        self.w_proj = nn.Parameter(torch.empty(config.n_experts, H, config.n_embd))
        self.shared_fc = Linear(config.n_embd, H * config.n_shared, bias=False) if config.n_shared > 0 else None
        self.shared_proj = Linear(H * config.n_shared, config.n_embd, bias=False) if config.n_shared > 0 else None
        self.register_buffer("route_bias", torch.zeros(config.n_experts)) # persistent: load balance state belongs in the checkpoint

    # NOTE: no torch.compiler.disable here. Since the padded-dispatch rewrite every
    # tensor in this path has a static shape (P is a fixed upper bound), so compile
    # only graph-breaks at the _grouped_mm custom Function instead of fragmenting
    # around the whole MoE layer. Validated by preflight; revert this commit if a
    # recompile storm appears on a new torch version.
    def forward(self, x):
        B, T, C = x.size()
        xf = x.view(-1, C)
        N = xf.size(0)
        router_logits = self.router(xf).float() # (N, E)
        if self.router_affinity == "sigmoid":
            affinity = torch.sigmoid(router_logits)
        else: # sqrtsoftplus (DSv4): unbounded positive affinity, no saturation
            affinity = F.softplus(router_logits).sqrt()
        _, topi = (affinity + self.route_bias).topk(self.n_topk, dim=-1) # (N, K), bias steers selection only
        gates = affinity.gather(-1, topi)
        gates = gates / gates.sum(-1, keepdim=True) # normalize over the selected experts
        if self.training:
            with torch.no_grad():
                load = torch.zeros(self.n_experts, device=xf.device)
                load.scatter_add_(0, topi.reshape(-1), torch.ones(topi.numel(), device=xf.device))
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(load) # keep route_bias identical across ranks
                err = load / load.sum() - 1.0 / self.n_experts
                self.route_bias -= self.bias_update_rate * torch.sign(err)
        # Dispatch: sort token-expert pairs by expert, one grouped GEMM per projection.
        # Each expert's block is padded to a multiple of 16 tokens with zero rows: the
        # grouped-GEMM wgrad kernel requires 16-byte-aligned group sizes, and zero rows
        # contribute nothing to outputs or grads. Buffer is statically sized and offsets
        # stay on the GPU: no CPU syncs. Empty experts still receive (zero) grads - Muon
        # requires grads on all params, and early routing can starve an expert for a
        # whole accum cycle.
        K = self.n_topk
        flat_topi = topi.reshape(-1) # (N*K)
        order = flat_topi.argsort()
        sorted_e = flat_topi[order]
        # scatter_add into a fixed buffer rather than bincount: bincount's output
        # shape is data-dependent, so dynamo cannot trace it and gives up on the
        # whole MoE forward ("Skipping the function and falling back to eager").
        # We already know the size is n_experts, so nothing here is actually dynamic.
        counts = torch.zeros(self.n_experts, dtype=torch.long, device=xf.device)
        counts.scatter_add_(0, flat_topi, torch.ones_like(flat_topi))
        pcounts = ((counts + 15) // 16) * 16
        poffs = pcounts.cumsum(0).to(torch.int32)
        cum0 = counts.cumsum(0) - counts # start of each expert's block in sorted order
        pstart = pcounts.cumsum(0) - pcounts # start of each expert's block in padded order
        within = torch.arange(N * K, device=xf.device) - cum0.gather(0, sorted_e)
        pos = pstart.gather(0, sorted_e) + within # padded slot of each sorted token
        src = torch.arange(N, device=xf.device).repeat_interleave(K)[order]
        P = N * K + 16 * self.n_experts # static upper bound on padded rows
        # D1: permute by gathering through an index map instead of moving the
        # activations three times. Pad slots and the unused tail point at row 0 and
        # are zeroed after the gather (see _GatherPermute for the autograd subtlety).
        # sum(pcounts) <= N*K + 15*E < P, so row P-1 is never inside any expert's
        # block: it is a safe sink for the masked-off lanes of pad_idx.
        gidx = torch.zeros(P, dtype=torch.long, device=xf.device)
        gidx.index_copy_(0, pos, src)
        r16 = torch.arange(16, device=xf.device)
        pad_idx = torch.where(r16 < (pcounts - counts).unsqueeze(1),
                              (pstart + counts).unsqueeze(1) + r16,
                              torch.full_like(counts[:1], P - 1).unsqueeze(1))
        # inverse permutation: perm[n*K + k] = padded slot of pair (n, k). Used by
        # both the output combine below and _GatherPermute's backward.
        perm = torch.zeros(N * K, dtype=torch.long, device=xf.device)
        perm.index_copy_(0, order, pos)
        xs = _GatherPermute.apply(xf, gidx, perm, pad_idx.reshape(-1), K)
        h = F.grouped_mm(xs, self.w_fc.to(xf.dtype), offs=poffs)
        h = F.relu(h).square()
        out = F.grouped_mm(h, self.w_proj.to(xf.dtype), offs=poffs)
        # D2: gather back through the inverse permutation so each token's K expert
        # outputs land adjacent, then reduce over K. Replaces a gather + an atomic
        # index_add_ (and drops the gate permutation: gates is already in (N,K) order).
        y = (out.index_select(0, perm).view(N, K, C) * gates.unsqueeze(-1).to(xf.dtype)).sum(1)
        if self.shared_fc is not None:
            y = y + self.shared_proj(F.relu(self.shared_fc(xf)).square())
        return y.view(B, T, C)


class MHCLayer(nn.Module):
    """
    One layer's manifold-constrained hyper-connection (DSv4 mHC).

    ⚠️ REDUCED IMPLEMENTATION — NOT a faithful reproduction. The A3 run using
    this (2026-07-24) is VOID as a test of mHC/hyper-connections; see
    DESIGN.md "METHODOLOGY RULE". Differences from the source design
    (Hyper-Connections, arXiv 2409.19606; DSv4 mHC):
      - single softmax read + single dynamic mixing term (source: separate
        depth-connection and width-connection matrices, per-stream learnable scales)
      - typically run at n=2 here; source's headline gains are at n=4
      - replaces nanochat's tuned resid/x0 lambdas, so the baseline is already
        enhanced (source compares against a *plain* residual)
    Before trusting any result from this module, implement DHC faithfully and
    compare against a plain-residual baseline.

    Maintains n parallel residual streams. The n x n stream-mixing matrix is
    projected onto (approximately) doubly stochastic matrices by Sinkhorn-Knopp,
    so the residual transform is non-expansive. Logits are static + a per-token
    input-dependent part (zero-init: training starts at the static point).
    Per-token dynamics keep the mix causal - never pool over time here.
    At init everything is uniform, streams stay identical, and the trunk is
    exactly a plain residual network (replaces resid/x0 lambdas when active).
    """
    def __init__(self, config):
        super().__init__()
        n = config.n_streams
        self.n = n
        self.sinkhorn_iters = config.mhc_sinkhorn_iters
        self.dyn_channels = 16
        self.mix_logits = nn.Parameter(torch.zeros(n, n))
        self.read_logits = nn.Parameter(torch.zeros(n))
        self.write_gate = nn.Parameter(torch.ones(n))
        self.dyn = Linear(self.dyn_channels, n * n, bias=False)

    def forward(self, X, x_in):
        # X: (B, T, n, C) streams; x_in: (B, T, C) the read vector (for the dynamic part)
        B, T, _, C = X.size()
        dyn_logits = self.dyn(x_in[..., :self.dyn_channels]).float() # (B, T, n*n)
        logits = self.mix_logits.float().view(1, 1, self.n, self.n) + dyn_logits.view(B, T, self.n, self.n)
        M = torch.exp(logits)
        for _ in range(self.sinkhorn_iters): # fixed iteration count: compile-friendly, no data-dependent control flow
            M = M / M.sum(dim=-1, keepdim=True)
            M = M / M.sum(dim=-2, keepdim=True)
        return torch.einsum('btij,btjc->btic', M.to(X.dtype), X)

    def read(self, X):
        r = F.softmax(self.read_logits.float(), dim=0).to(X.dtype)
        return torch.einsum('btnc,n->btc', X, r)


class AttnRes(nn.Module):
    """Attention Residuals (Kimi Team, arXiv 2603.15031), Full variant.

    Replaces `h_l = h_{l-1} + f_{l-1}(h_{l-1})` with a softmax attention over the
    outputs of every earlier sublayer:

        h_l = sum_i alpha_{i->l} * v_i,  alpha = softmax_i( q_l . RMSNorm(k_i) )
        k_i = v_i = f_i(h_i)  (v_0 = the token embedding)

    One instance per *sublayer* (attention and MLP each get their own), so a
    d-layer model has 2d + 1 of these counting the final aggregation.

    Two details the paper is emphatic about, both easy to get wrong:
    - the pseudo-query is initialized to ZERO, so at init every source gets equal
      weight and the layer starts as a plain average (they report this prevents
      training volatility);
    - RMSNorm applies to the key path only. The values are summed un-normalized,
      which is what stops large-magnitude layers from dominating the weights
      without also rescaling their contribution.

    The paper's per-layer RMSNorm carries a learnable gain. We use nanochat's
    parameterless norm instead, which is equivalent rather than a simplification:
    a diagonal gain g would appear only as q.(g * RMSNorm(k)) = (q * g).RMSNorm(k),
    i.e. absorbed into the pseudo-query we are already learning.
    """
    def __init__(self, config):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(config.n_embd)) # zero-init is load-bearing

    def forward(self, sources, keys):
        # `keys` are the RMSNormed sources, normed once when each source is produced.
        # A source's key is identical for every later sublayer that reads it, so
        # normalizing inside here would redo the same work O(L^2) times instead of
        # O(L) — worth ~40% of this layer's cost at d12. Same value either way.
        #
        # NOTE: the first instance sees a single source (the embedding), so its
        # softmax is identically 1 and its pseudo-query is dead. We deliberately do
        # NOT short-circuit that case: running the softmax anyway keeps the grad at
        # zero rather than None, and a None grad crashes the fused AdamW step.
        q = self.q.to(sources[0].dtype)
        # Score against the keys rather than stacking the values: a stacked
        # (S,B,T,C) buffer would be ~1GB per call at d12/bs16, and only the
        # (S,B,T) scores are actually needed at full width.
        logits = torch.stack([(k * q).sum(-1) for k in keys], dim=0).float()
        w = logits.softmax(dim=0).to(sources[0].dtype) # softmax over the depth axis
        out = w[0].unsqueeze(-1) * sources[0]
        for i in range(1, len(sources)):
            out = out + w[i].unsqueeze(-1) * sources[i]
        return out


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        use_moe = config.n_experts > 0 and layer_idx >= config.moe_first_dense
        self.mlp = MoEMLP(config) if use_moe else MLP(config)

    # AttnRes needs the sublayer outputs on their own (it supplies its own
    # cross-layer mixing in place of the residual adds below).
    def attn_out(self, x, ve, cos_sin, window_size, kv_cache):
        return self.attn(norm(x), ve, cos_sin, window_size, kv_cache)

    def mlp_out(self, x):
        return self.mlp(norm(x))

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # mHC: per-layer stream mixers + final read-out (replaces resid/x0 lambdas when active)
        self.mhc = nn.ModuleList([MHCLayer(config) for _ in range(config.n_layer)]) if config.n_streams > 1 else None
        self.final_read_logits = nn.Parameter(torch.zeros(config.n_streams)) if config.n_streams > 1 else None
        # 2 per block (before attention, before MLP) + 1 for the final aggregation,
        # which the paper specifies also reads all sources rather than just the last.
        assert not (config.attn_res and config.n_streams > 1), "AttnRes and mHC both replace the residual stream"
        self.attn_res = nn.ModuleList([AttnRes(config) for _ in range(2 * config.n_layer + 1)]) if config.attn_res else None
        # MTP (DeepSeek-V3): sequential modules predicting t+2, t+3, ... Each module is a
        # transformer block (layer_idx=n_layer => no value-embedding gate) plus a projection
        # combining the previous depth's hidden with the look-ahead token embedding. wte and
        # lm_head are shared with the main model (not re-created here).
        if config.n_mtp > 0:
            self.mtp_blocks = nn.ModuleList([Block(config, config.n_layer) for _ in range(config.n_mtp)])
            self.mtp_proj = nn.ModuleList([Linear(2 * config.n_embd, config.n_embd, bias=False) for _ in range(config.n_mtp)])
        else:
            self.mtp_blocks = None
            self.mtp_proj = None
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        init_blocks = list(self.transformer.h)
        if self.mtp_blocks is not None:
            init_blocks += list(self.mtp_blocks) # MTP blocks init identically to trunk blocks
        for block in init_blocks:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if isinstance(block.mlp, MoEMLP):
                torch.nn.init.uniform_(block.mlp.router.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.route_bias)
                torch.nn.init.uniform_(block.mlp.w_fc, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.w_proj)
                if block.mlp.shared_fc is not None:
                    torch.nn.init.uniform_(block.mlp.shared_fc.weight, -s * 0.4, s * 0.4)
                    torch.nn.init.zeros_(block.mlp.shared_proj.weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # MTP projection matrices M_k (concat of two normed d-vectors -> d)
        if self.mtp_proj is not None:
            for proj in self.mtp_proj:
                torch.nn.init.uniform_(proj.weight, -s * 0.4, s * 0.4)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # mHC: uniform mixing/read, unit write, zero dynamics => exactly a plain residual net at init
        if self.mhc is not None:
            for layer in self.mhc:
                torch.nn.init.zeros_(layer.mix_logits)
                torch.nn.init.zeros_(layer.read_logits)
                torch.nn.init.ones_(layer.write_gate)
                torch.nn.init.zeros_(layer.dyn.weight)
            torch.nn.init.zeros_(self.final_read_logits)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        For MoE, only the top-k routed experts touch each token, so inactive expert
        params are excluded (this is a FLOPs count, not a storage count).
        MTP modules are excluded: they run in training only and are discarded at
        inference, so this count reflects the deployed model. (Consequence: on MTP
        runs, per-step MFU under-reads, since real training FLOPs include the MTP
        blocks but this estimate does not.)
        """
        mtp_ids = set()
        if self.mtp_blocks is not None:
            for m in list(self.mtp_blocks.modules()) + list(self.mtp_proj.modules()):
                mtp_ids.add(id(m))
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear) and id(m) not in mtp_ids)
        for block in self.transformer.h:
            if isinstance(block.mlp, MoEMLP):
                expert_params = block.mlp.w_fc.numel() + block.mlp.w_proj.numel() # not Linears: add active fraction
                matmul_params += round(expert_params * block.mlp.n_topk / block.mlp.n_experts)
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        mtp = 0
        if self.mtp_blocks is not None:
            mtp = sum(p.numel() for p in self.mtp_blocks.parameters()) + sum(p.numel() for p in self.mtp_proj.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        if self.mhc is not None:
            scalars += sum(p.numel() for p in self.mhc.parameters()) + self.final_read_logits.numel()
        if self.attn_res is not None:
            # one d-vector pseudo-query per sublayer: vectors, not matrices
            scalars += sum(p.numel() for p in self.attn_res.parameters())
        total = wte + value_embeds + lm_head + transformer_matrices + mtp + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'mtp': mtp,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        # MoE routers go to AdamW, not Muon: orthogonalizing a routing matrix distorts the affinities.
        # MTP blocks/projections are ordinary matrices and get the same treatment as trunk blocks.
        block_named = list(self.transformer.h.named_parameters())
        if self.mtp_blocks is not None:
            block_named += list(self.mtp_blocks.named_parameters()) + list(self.mtp_proj.named_parameters())
        router_params = [p for n, p in block_named if "router" in n]
        matrix_params = [p for n, p in block_named if "router" not in n]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        mhc_params = (list(self.mhc.parameters()) + [self.final_read_logits]) if self.mhc is not None else []
        attn_res_params = list(self.attn_res.parameters()) if self.attn_res is not None else []
        assert len(list(self.parameters())) == len(matrix_params) + len(router_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(mhc_params) + len(attn_res_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if self.mhc is None and self.attn_res is None:
            param_groups.append(dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05))
            param_groups.append(dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0))  # higher beta1 for x0
        else:
            # mHC and AttnRes both replace the resid/x0 machinery: those params are
            # unused in forward, would get grad=None, and the fused AdamW step cannot
            # handle that. Freeze them.
            for p in resid_params + x0_params:
                p.requires_grad_(False)
        if router_params:
            param_groups.append(dict(kind='adamw', params=router_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if mhc_params:
            # conservative lr: the mixing matrix steers every residual in the network
            param_groups.append(dict(kind='adamw', params=mhc_params, lr=0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if attn_res_params:
            # pseudo-queries: d-vectors, not matrices, so AdamW rather than Muon
            # (orthogonalizing a vector is meaningless). No weight decay — decaying
            # them back toward zero would pin the layer at a uniform average.
            param_groups.append(dict(kind='adamw', params=attn_res_params, lr=scalar_lr * 0.04, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def lm_head_logits(self, h):
        """Project an already-normed hidden state to softcapped logits. DeepSeek-V3
        shares this head between the trunk and every MTP depth, so it lives here
        rather than as a closure inside forward()."""
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        z = self.lm_head(h)[..., :self.config.vocab_size].float()
        return softcap * torch.tanh(z / softcap)

    @torch.inference_mode()
    def mtp_step(self, h_prev, idx_ahead, kv_cache, depth=0):
        """Run one MTP module at inference, mirroring the training recipe.

        h_prev: (B, T, C) hidden from the previous depth (h^0 for depth 0) at
        consecutive positions; idx_ahead: (B, T) the token one position ahead of
        each. Returns (logits, h_k), where logits predict the token *two* ahead.
        kv_cache is the MTP module's own cache and must be kept dense over token
        positions so attention matches what the module saw during training. Pass
        kv_cache=None to run over a whole sequence exactly as training does.
        """
        T0, T = (0 if kv_cache is None else kv_cache.get_pos()), h_prev.size(1)
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        emb_ahead = norm(self.transformer.wte(idx_ahead).to(h_prev.dtype))
        hin = self.mtp_proj[depth](torch.cat([norm(h_prev), emb_ahead], dim=-1))
        h_k = self.mtp_blocks[depth](hin, None, cos_sin, self.window_sizes[-1], kv_cache)
        return self.lm_head_logits(norm(h_k)), h_k

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_hidden=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Chunk: smear positions 1+ against their in-chunk predecessor, same
                # as training. Position 0 of the chunk has no in-chunk predecessor, so
                # it smears against the cached embedding from the previous chunk (None
                # only on a true prefill, where position 0 has no predecessor at all).
                # Without this, a multi-token decode step diverges from a single-token
                # one — which would silently corrupt speculative decoding.
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x_rest = x[:, 1:] + gate * x[:, :-1]
                x_first = x[:, :1]
                if x_pre_smear is not None:
                    gate0 = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :1, :24]))
                    x_first = x_first + gate0 * x_pre_smear
                x = torch.cat([x_first, x_rest], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        if self.mhc is not None:
            # mHC trunk: n parallel streams, doubly-stochastic mixing per layer.
            # Replaces the resid/x0 lambda machinery (mixing subsumes both roles).
            X = x.unsqueeze(2).expand(-1, -1, self.config.n_streams, -1).contiguous()
            for i, block in enumerate(self.transformer.h):
                x_in = self.mhc[i].read(X)
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                y = block(x_in, ve, cos_sin, self.window_sizes[i], kv_cache) - x_in # block adds x_in internally; extract the delta
                X = self.mhc[i](X, x_in) + self.mhc[i].write_gate.to(X.dtype).view(1, 1, -1, 1) * y.unsqueeze(2)
                if i == backout_layer:
                    x_backout = X.mean(dim=2)
            x = torch.einsum('btnc,n->btc', X, F.softmax(self.final_read_logits.float(), dim=0).to(X.dtype))
        elif self.attn_res is not None:
            # AttnRes trunk: no running residual. Every sublayer reads a softmax
            # mixture of all previous sublayer outputs, starting from the embedding.
            sources, keys = [x0], [norm(x0)]

            def add(v): # each source is normed once, here, and reused as a key
                sources.append(v)
                keys.append(norm(v))

            for i, block in enumerate(self.transformer.h):
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                h = self.attn_res[2 * i](sources, keys)
                if i == backout_layer + 1:
                    x_backout = h # the state entering this block == the state after backout_layer
                add(block.attn_out(h, ve, cos_sin, self.window_sizes[i], kv_cache))
                add(block.mlp_out(self.attn_res[2 * i + 1](sources, keys)))
            x = self.attn_res[-1](sources, keys)
        else:
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                if i == backout_layer:
                    x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        h0 = norm(x) # main-model hidden feeding the shared head (= h^0 for MTP)

        # Forward the lm_head (compute logits)
        shared_head = self.lm_head_logits # norm already applied by caller

        if targets is None:
            # inference: just return the logits (MTP modules are unused for plain
            # decoding; self-speculative decoding asks for h0 via return_hidden)
            logits = shared_head(h0) # (B, T, vocab_size)
            return (logits, h0) if return_hidden else logits

        # The fused path never materializes the (B*T, vocab) fp32 logits, which are
        # the largest tensor in the step and are built twice once MTP is on. It only
        # implements the mean reduction; bpb eval asks for 'none' and takes the
        # plain path below.
        fused = self.config.fused_ce and loss_reduction == 'mean'
        head_w = self.lm_head.weight[:self.config.vocab_size]

        def head_loss(h, tgt):
            if fused:
                return linear_cross_entropy(h.view(-1, h.size(-1)), head_w.to(h.dtype),
                                            tgt.reshape(-1), softcap=15.0, ignore_index=-1)
            lg = shared_head(h)
            return F.cross_entropy(lg.view(-1, lg.size(-1)), tgt.reshape(-1),
                                   ignore_index=-1, reduction=loss_reduction)

        loss = head_loss(h0, targets)

        # Multi-Token Prediction (DeepSeek-V3): sequential modules predict t+2, t+3, ...
        # Each depth k combines the previous depth's hidden with the embedding of the
        # token k ahead, runs a transformer block, and predicts the token k+1 ahead via
        # the shared head. Training-only auxiliary loss; excluded from bpb eval (which
        # runs under model.eval() with reduction='none').
        if self.mtp_blocks is not None and self.training and loss_reduction == 'mean':
            mtp_losses = []
            h_prev = h0
            emb = norm(self.transformer.wte(idx).to(h0.dtype))               # Emb(t_i), shared across depths
            for k in range(1, self.config.n_mtp + 1):
                emb_ahead = F.pad(emb[:, k:], (0, 0, 0, k))                    # Emb(t_{i+k}), tail zero-padded
                hin = self.mtp_proj[k-1](torch.cat([norm(h_prev), emb_ahead], dim=-1))
                h_k = self.mtp_blocks[k-1](hin, None, cos_sin, self.window_sizes[-1], None)
                tgt_k = F.pad(targets[:, k:], (0, k), value=-1)               # predict t_{i+k+1}
                mtp_losses.append(head_loss(norm(h_k), tgt_k))
                h_prev = h_k
            loss = loss + self.config.mtp_weight * torch.stack(mtp_losses).mean()

        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
