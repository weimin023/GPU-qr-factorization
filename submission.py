#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return torch.geqrf(data)
