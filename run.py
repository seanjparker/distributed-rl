import torch.multiprocessing as mp
from algos.torch.ppo_ddp import ppo as torch_ddp_ppo
from algos.torch.ppo_mpi import ppo as torch_mpi_ppo
from algos.tf.ppo_mpi import ppo as tf_mpi_ppo
from algos.common.mpi import mpi_fork

import argparse
import psutil


def run_torch_ppo(workers, epochs):
    mpi_fork(workers)
    torch_mpi_ppo(workers, epochs)


def run_tf_ppo(workers, epochs):
    mpi_fork(workers)
    tf_mpi_ppo(workers, epochs)


def run_experiment(framework, workers, epochs):
    print(f'Running experiment with framework: {framework}, workers: {workers}')
    if framework == 'tf':
        run_tf_ppo(workers, epochs)
    elif framework == 'torch':
        run_torch_ppo(workers, epochs)
    elif framework == 'ray':
        return


parser = argparse.ArgumentParser()
parser.add_argument('--fw', type=str, choices=['torch', 'tf', 'ray'], help='choice of framework')
parser.add_argument('--w', type=int, help='number of workers')
parser.add_argument('--b', type=str, choices=['mpi', 'builtin'], help='choice of backend communication type')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')


args = parser.parse_args()
if args.b == 'mpi' and args.fw == 'ray':
    raise Exception('cannot use MPI with Ray engine')

if __name__ == '__main__':
    av_cpu = psutil.cpu_count()
    if args.w > av_cpu:
        raise Exception(f'want {args.w} cpu, have {av_cpu}')

    run_experiment(framework=args.fw, workers=args.w, epochs=args.epochs)

    # if mpi_proc_id() == 0:
    #    timestamp = datetime.now().strftime("%H:%M:%S")
    #    torch.save(model.state_dict(), f'{ROOT_DIR}/models/{timestamp}_torchppo.pt')

