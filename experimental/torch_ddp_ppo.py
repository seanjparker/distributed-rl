import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import gym
from scipy import signal

import os

# Hyperparameter definitions
num_workers = 1  # same thing as 'world size'
pi_lr = 3e-4
v_lr = 1e-3
epochs = 50  # 50
steps_per_epoch = 4000 // num_workers
max_ep_len = 1000
train_pi_iters = 80
train_v_iters = 80
gamma = 0.99
lam = 0.97
clip_ratio_offset = 0.2


def setup(rank, world_size):
    # Only works on Mac/Linux
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Actor(nn.Module):
    def __init__(self, obs_dimensions, act_dimensions):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dimensions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dimensions),
            nn.Tanh()
        )
        self.get_cat_dist = self.get_distribution

    def get_distribution(self, obs):
        return Categorical(logits=self.actor(obs))

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions
        pi = self.get_cat_dist(obs)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class Critic(nn.Module):
    def __init__(self, obs_dimensions):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_dimensions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
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

    def forward(self, obs):
        with torch.no_grad():
            pi = self.pi.get_cat_dist(obs)
            action = pi.sample()
            logp_a = pi.log_prob(action)
            val = self.v(obs)

        return action.numpy(), val.numpy(), logp_a.numpy()

    # def get_action(self, obs):
    #    return self.step(obs)[0]


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def ppo(rank):
    print(f'Running basic DDP example on rank {rank}')
    setup(rank, num_workers)

    env = gym.make('LunarLander-v2')
    obs_space = env.observation_space
    act_space = env.action_space

    actor_critic = ActorCritic(obs_space, act_space).to('cpu')
    ac_ddp = DDP(actor_critic)

    obs_buf = np.zeros((steps_per_epoch, obs_space.shape[0]), dtype=np.float32)
    act_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    adv_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_boot_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    val_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    logp_buf = np.zeros(steps_per_epoch, dtype=np.float32)

    pi_optim = Adam(ac_ddp.pi.parameters(), lr=pi_lr)
    v_optim = Adam(ac_ddp.v.parameters(), lr=v_lr)

    def ppo_pi_loss(data):
        obs, action, advantage, old_logp = data['obs'], data['act'], data['adv'], data['logp']

        pi, logp = ac_ddp.pi(obs, action)
        pi_ratio = torch.exp(logp - old_logp)
        clipped_adv = torch.clamp(pi_ratio, 1 - clip_ratio_offset, 1 + clip_ratio_offset) * advantage
        pi_loss = -(torch.min(pi_ratio * advantage, clipped_adv)).mean()
        return pi_loss

    def ppo_v_loss(data):
        obs, adj_rewards = data['obs'], data['rew']
        return (ac_ddp.v(obs) - adj_rewards).pow(2).mean()

    def ppo_update():
        data = {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in dict(obs=obs_buf, act=act_buf, rew=rew_boot_buf, adv=adv_buf, logp=logp_buf).items()}

        for i in range(train_pi_iters):
            pi_optim.zero_grad()
            pi_loss = ppo_pi_loss(data)
            pi_loss.backward()
            pi_optim.step()

        for i in range(train_v_iters):
            v_optim.zero_grad()
            v_loss = ppo_v_loss(data)
            v_loss.backward()
            v_optim.step()

    obs, ep_reward, ep_len = env.reset(), 0, 0

    ep_reward_history = np.zeros(epochs, dtype=np.float32)
    for ep in range(epochs):
        for t in range(steps_per_epoch):
            action, val, logp_a = ac_ddp(torch.as_tensor(obs, dtype=torch.float32))

            new_obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_len += 1

            # buf.store(obs, action, reward, val, logp_a)
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
                    _, bootstrap_v, _ = ac_ddp.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    bootstrap_v = 0

                rews_aug = np.append(rew_buf[:], bootstrap_v)
                vals_aug = np.append(val_buf[:], bootstrap_v)

                # GAE-Lambda advantage calculation
                gae_deltas = rews_aug[:-1] + gamma * vals_aug[1:] - vals_aug[:-1]
                adv_buf[:] = discount_cumsum(gae_deltas, gamma * lam)

                # Computes rewards-to-go, to be targets for the value function
                rew_boot_buf[:] = discount_cumsum(rews_aug, gamma)[:-1]

                ep_reward_history[ep] = max(ep_reward_history[ep], ep_reward)
                obs, ep_reward, done = env.reset(), 0, 0

        ppo_update()
        print(f'device: {rank}, epoch: {ep + 1}, max reward: {ep_reward_history[ep]:.3f}')
    cleanup()


if __name__ == '__main__':
    mp.spawn(ppo, nprocs=num_workers, join=True)