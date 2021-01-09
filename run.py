import torch.multiprocessing as mp
from algos.torch.ppo_ddp import ppo


if __name__ == '__main__':
    mp.spawn(ppo, nprocs=2, join=True)
