import sys, types, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

task_module = types.ModuleType("task")
task_module.input_t = torch.Tensor
task_module.output_t = tuple
sys.modules["task"] = task_module

import qr_official
import submission as S
from cutlass.cute.runtime import from_dlpack

data = qr_official.generate_input(batch=4, n=128, cond=1, seed=99, case="dense")
batch, m, n = data.shape
nb = 64
k = min(m, n)

h_ref = data.clone()
tau = torch.zeros(batch, k, device=data.device, dtype=data.dtype)
j_start, j_end = 0, nb
S._panel_factor_apply_cutedsl_mvp(h_ref, tau, j_start, j_end)
t_ws = torch.empty(batch, nb, nb, device=data.device, dtype=data.dtype)
S._build_compact_wy_t_cutedsl(h_ref, tau, t_ws, j_start, j_end)
torch.cuda.synchronize()

panel = h_ref[:, j_start:m, j_start:j_end]
V = torch.tril(panel, diagonal=-1).clone()
eye = torch.eye(nb, device=data.device, dtype=data.dtype).unsqueeze(0)
V[:, 0:nb, 0:nb] = V[:, 0:nb, 0:nb] + eye
tau_p = tau[:, j_start:j_end]

def larft_forward(V, tau_p):
    b, mm, kk = V.shape
    T = torch.zeros(b, kk, kk, device=V.device, dtype=V.dtype)
    VtV = torch.bmm(V.transpose(-1,-2), V)
    for i in range(kk):
        T[:, i, i] = tau_p[:, i]
        if i > 0:
            col = -tau_p[:, i:i+1] * VtV[:, 0:i, i]
            T[:, 0:i, i] = torch.bmm(T[:, 0:i, 0:i], col.unsqueeze(-1)).squeeze(-1)
    return T

T_up = larft_forward(V, tau_p)
print("t_ws == T_up^T ?  max diff:", (t_ws - T_up.transpose(-1,-2)).abs().max().item())

# trailing update using T_up with correct application: I - V T_up V^T uses Z = T_up^T @ Y? test both
C = h_ref[:, j_start:m, j_end:n]
h_custom = h_ref.clone()
row_tiles = (m - j_start + 63)//64
y_ws = torch.empty(batch, nb, n-nb, (m+63)//64, device=data.device, dtype=data.dtype)
S.part5_apply_panel_wy_fused_update_cuda(from_dlpack(h_custom), from_dlpack(t_ws), from_dlpack(y_ws), batch, m, n, j_start, j_end, row_tiles)
torch.cuda.synchronize()
ref = h_custom[:, j_start:m, j_end:n]

Y = torch.bmm(V.transpose(-1,-2), C)
for label, Tuse in [("T_up", T_up), ("T_up^T", T_up.transpose(-1,-2))]:
    Z = torch.bmm(Tuse, Y)
    Cnew = C - torch.bmm(V, Z)
    print(label, "trailing diff vs custom:", (ref - Cnew).abs().max().item())
