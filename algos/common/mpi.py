from mpi4py import MPI
import os
import subprocess
import sys
import numpy as np

# -------- General MPI functions --------


def mpi_proc_id():
    return MPI.COMM_WORLD.Get_rank()


def mpi_num_procs():
    return MPI.COMM_WORLD.Get_size()


def mpi_op(x, op=MPI.SUM):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    MPI.COMM_WORLD.Allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_avg(x):
    return mpi_op(x) / mpi_num_procs()


def mpi_avg_scalar(x):
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_op([np.sum(x), x.size])
    mean = global_sum / global_n
    return mean


def mpi_fork(n):
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', IN_MPI='1')
        args = ['mpirun', '-hostfile', '/etc/hosts', '-np', str(n), '--use-hwthread-cpus']
        args += [sys.executable] + sys.argv
        print(args)
        subprocess.check_call(args, env=env)
        sys.exit()

# -------- Torch-specific MPI functions --------


def torch_mpi_sync_params(module):
    if mpi_num_procs() == 1:
        return
    for p in module.parameters():
        MPI.COMM_WORLD.Bcast(p.data.numpy(), root=0)


def torch_mpi_avg_grads(module):
    if mpi_num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]
