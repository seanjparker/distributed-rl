import os
import subprocess
import sys
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
from mpi4py import MPI
from scipy import signal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from algos.common.util import EpochRecorder

from defs import ROOT_DIR

# Hyperparameter definitions
num_workers = 4
pi_lr = 3e-4
v_lr = 1e-3
epochs = 50
steps_per_epoch = 4000 // num_workers
max_ep_len = 1000
train_pi_iters = 80
train_v_iters = 80
gamma = 0.99
lam = 0.97
clip_ratio_offset = 0.2


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


def mpi_avg_grads(module):
    if mpi_num_procs() == 1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def mpi_avg_scalar(x):
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_op([np.sum(x), x.size])
    mean = global_sum / global_n
    return mean


def mpi_sync_params(module):
    if mpi_num_procs() == 1:
        return
    for p in module.parameters():
        MPI.COMM_WORLD.Bcast(p.data.numpy(), root=0)


def mpi_fork(n):
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', IN_MPI='1')
        args = ['mpirun', '-np', str(n), '--use-hwthread-cpus']
        args += [sys.executable] + sys.argv
        print(args)
        subprocess.check_call(args, env=env)
        sys.exit()


def torch_mpi_init():
    seed = 1337 * (mpi_proc_id() + 1)
    torch.manual_seed(seed)
    np.random.seed(seed)


class Actor(nn.Module):
    def __init__(self, obs_dimensions, act_dimensions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dimensions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dimensions),
            nn.Identity()
        )

    def get_distribution(self, obs):
        return Categorical(logits=self.actor(obs))

    def forward(self, obs, act=None):
        # Produce action distributions for given observations
        # Compute log likelihood of given actions under those distributions
        pi = self.get_distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class Critic(nn.Module):
    def __init__(self, obs_dimensions):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dimensions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Identity()
        )

    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_space, act_space):
        super().__init__()

        obs_dimensions = obs_space.shape[0]

        # policy network
        self.pi = Actor(obs_dimensions, act_space.n)

        # value network
        self.v = Critic(obs_dimensions)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi.get_distribution(obs)
            action = pi.sample()
            logp_a = pi.log_prob(action)
            val = self.v(obs)

        return action.numpy(), val.numpy(), logp_a.numpy()

    def get_action(self, obs):
        return self.step(obs)[0]


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def ppo():
    rank = mpi_proc_id()
    torch_mpi_init()

    reward_rec = EpochRecorder(rank, 'mean reward')

    env = gym.make('LunarLander-v2')
    obs_space = env.observation_space
    act_space = env.action_space

    actor_critic = ActorCritic(obs_space, act_space)

    mpi_sync_params(actor_critic)

    obs_buf = np.zeros((steps_per_epoch, obs_space.shape[0]), dtype=np.float32)
    act_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    adv_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_boot_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    val_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    logp_buf = np.zeros(steps_per_epoch, dtype=np.float32)

    pi_optim = Adam(actor_critic.pi.parameters(), lr=pi_lr)
    v_optim = Adam(actor_critic.v.parameters(), lr=v_lr)

    def ppo_pi_loss(data):
        data_obs, data_action, advantage, old_logp = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = actor_critic.pi(data_obs, data_action)
        pi_ratio = torch.exp(logp - old_logp)
        clipped_adv = torch.clamp(pi_ratio, 1 - clip_ratio_offset, 1 + clip_ratio_offset) * advantage
        pi_loss = -(torch.min(pi_ratio * advantage, clipped_adv)).mean()
        return pi_loss

    def ppo_v_loss(data):
        obs, adj_rewards = data['obs'], data['rew']
        return (actor_critic.v(obs) - adj_rewards).pow(2).mean()

    def ppo_update():
        # TODO: Advantage normalisation trick
        data = {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in dict(obs=obs_buf, act=act_buf, rew=rew_boot_buf, adv=adv_buf, logp=logp_buf).items()}

        for i in range(train_pi_iters):
            pi_optim.zero_grad()
            pi_loss = ppo_pi_loss(data)
            pi_loss.backward()
            mpi_avg_grads(actor_critic.pi)
            pi_optim.step()

        for i in range(train_v_iters):
            v_optim.zero_grad()
            v_loss = ppo_v_loss(data)
            v_loss.backward()
            mpi_avg_grads(actor_critic.v)
            v_optim.step()

    obs, ep_reward, ep_len = env.reset(), 0, 0
    ep_reward_history = [list() for _ in range(epochs)]
    for ep in range(epochs):
        for t in range(steps_per_epoch):
            action, val, logp_a = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))

            new_obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_len += 1

            obs_buf[t] = obs
            act_buf[t] = action
            rew_buf[t] = reward
            val_buf[t] = val
            logp_buf[t] = logp_a

            obs = new_obs

            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, bootstrap_v, _ = actor_critic.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    bootstrap_v = 0

                rews_aug = np.append(rew_buf[:], bootstrap_v)
                vals_aug = np.append(val_buf[:], bootstrap_v)

                # GAE-Lambda advantage calculation
                gae_deltas = rews_aug[:-1] + gamma * vals_aug[1:] - vals_aug[:-1]
                adv_buf[:] = discount_cumsum(gae_deltas, gamma * lam)

                # Computes rewards-to-go, to be targets for the value function
                rew_boot_buf[:] = discount_cumsum(rews_aug, gamma)[:-1]

                ep_reward_history[ep].append(ep_reward)
                obs, ep_reward, done = env.reset(), 0, 0

        ppo_update()

        # Average the minibatch rewards across all processes
        rank_rwd_mean = np.mean(ep_reward_history[ep])
        avg_rwd = mpi_avg_scalar(rank_rwd_mean)

        # Record the avg reward so we can save it to a file later
        if rank == 0:
            reward_rec.store(avg_rwd.item())
            print(f'proc id: {rank}, epoch: {ep + 1}, mean reward: {avg_rwd:.3f}')

    # Training complete, dump the data to JSON
    if mpi_proc_id() == 0:
        reward_rec.dump()

    return actor_critic


if __name__ == '__main__':
    mpi_fork(num_workers)
    model = ppo()
    if mpi_proc_id() == 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        torch.save(model.state_dict(), f'{ROOT_DIR}/models/{timestamp}_torchppo.pt')


