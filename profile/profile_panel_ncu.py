import argparse
import sys
import types
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--nb", type=int, default=64)
    parser.add_argument("--cond", type=int, default=1)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--case", default="dense")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    data = qr_official.generate_input(
        batch=args.batch,
        n=args.n,
        cond=args.cond,
        seed=args.seed,
        case=args.case,
    )
    tau = torch.zeros(args.batch, args.n, device=data.device, dtype=data.dtype)
    j_start = 0
    j_end = min(args.nb, args.n)

    for _ in range(args.iters):
        h = data.clone()
        tau.zero_()
        S._panel_factor_apply_cutedsl_mvp(h, tau, j_start, j_end)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
