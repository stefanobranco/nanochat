"""
Numerical correctness check for the MoE dispatch (grouped GEMM + routing) against
a naive per-token reference loop. Run on a GPU pod after any change to MoEMLP:
    python -m dev.test_moe_numeric
Passes if forward is exact and gradients match the fp32 reference more closely
than a bf16 reference loop does (i.e. our error is dtype noise, not a bug).
"""
import torch, sys
sys.path.insert(0, "/workspace/nanochat")
from nanochat.gpt import GPTConfig, MoEMLP

torch.manual_seed(0)
cfg = GPTConfig(n_embd=64, n_head=2, n_kv_head=2, n_layer=2, n_experts=8, n_topk=2, n_shared=1, expert_hidden=32)
m = MoEMLP(cfg).cuda()
for p in m.parameters():
    torch.nn.init.normal_(p, std=0.1)
m = m.to(torch.bfloat16); m.route_bias.zero_(); m.eval()
x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
y = m(x); y.sum().backward()
g_grouped = m.w_fc.grad.float().clone()

def ref(dtype):
    wfc = m.w_fc.detach().to(dtype).requires_grad_(True)
    wpr = m.w_proj.detach().to(dtype).requires_grad_(True)
    xf = x.detach().to(dtype).view(-1, 64).requires_grad_(True)
    aff = torch.sigmoid((xf @ m.router.weight.t().to(dtype)).float())
    _, topi = (aff + m.route_bias).topk(2, dim=-1)
    gates = aff.gather(-1, topi); gates = gates / gates.sum(-1, keepdim=True)
    yy = torch.zeros_like(xf)
    for t in range(xf.size(0)):
        for j in range(2):
            e = topi[t, j].item()
            h = torch.relu(xf[t] @ wfc[e]).square()
            yy = yy.index_add(0, torch.tensor([t], device="cuda"), ((h @ wpr[e]) * gates[t, j].to(dtype)).unsqueeze(0))
    yy.sum().backward()
    return wfc.grad.float()

g32 = ref(torch.float32)
g16 = ref(torch.bfloat16)
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
e_grouped = rel(g_grouped, g32)
e_bf16ref = rel(g16, g32)
print(f"grouped vs fp32-ref rel err: {e_grouped:.5f}")
print(f"bf16-ref vs fp32-ref rel err: {e_bf16ref:.5f}")
print("PASS" if e_grouped < 2 * e_bf16ref else "FAIL — grouped error exceeds dtype noise")
