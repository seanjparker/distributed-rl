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

import os, subprocess, sys

# Hyperparameter definitions
from algos.common.util import EpochRecorder

num_workers = 2  # same thing as 'world size'
pi_lr = 3e-4
v_lr = 1e-3
epochs = 5  # 50
steps_per_epoch = 4000 // num_workers
max_ep_len = 1000
train_pi_iters = 80
train_v_iters = 80
gamma = 0.99
lam = 0.97
clip_ratio_offset = 0.2


def ddp_avg_reward(rank, reward):
    # Get the average reward across processes, returns to only rank 0
    size = float(dist.get_world_size())
    summaries = torch.tensor(reward, requires_grad=False, device='cpu')
    dist.reduce(summaries, dst=0)
    if rank == 0:
        summaries = summaries / size
        return summaries
    return None


def setup(rank, world_size):
    # Set a random seed based on the rank
    seed = 1337 * (rank + 1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Only works on Mac/Linux
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # Initialize the process group
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class Actor(nn.Module):
    def __init__(self, obs_dimensions, act_dimensions):
        super().__init__()
        # Define the actor NN, output one value per action in action space
        self.actor = nn.Sequential(
            nn.Linear(obs_dimensions, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dimensions),
            nn.Identity()
        )
        
    def get_distribution(self, obs):
        # Using the raw logits from the actor gets the Categorical distribution
        return Categorical(logits=self.actor(obs))
    
    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions
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
    def __init__(self, actor: DDP, critic: DDP):
        super().__init__()
        
        # policy network
        self.pi = actor
        
        # value network
        self.v = critic

    def forward(self, obs):
        with torch.no_grad():
            pi = self.pi.module.get_distribution(obs)
            action = pi.sample()
            logp_a = pi.log_prob(action)
            val = self.v(obs)

        return action.numpy(), val.numpy(), logp_a.numpy()
    
    def get_action(self, obs):
        return self(obs)[0]


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def ppo(rank):
    print(f'Running PPO using DDP -- rank {rank}')
    setup(rank, num_workers)
    
    env = gym.make('LunarLander-v2')
    obs_space = env.observation_space
    act_space = env.action_space

    obs_dimensions = obs_space.shape[0]

    actor = Actor(obs_dimensions, act_space.n)
    critic = Critic(obs_dimensions)
    actor_ddp = DDP(actor)
    critic_ddp = DDP(critic)

    actor_critic = ActorCritic(actor_ddp, critic_ddp)

    reward_rec = EpochRecorder(rank, 'reward')
    
    obs_buf = np.zeros((steps_per_epoch, obs_dimensions), dtype=np.float32)
    act_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    adv_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_boot_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    val_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    logp_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    
    pi_optim = Adam(actor_ddp.parameters(), lr=pi_lr)
    v_optim = Adam(critic_ddp.parameters(), lr=v_lr)
    
    def ppo_pi_loss(data):
        # Extract the observation, action, advantage and prev_log_probs from buffer
        obs, action, advantage, old_logp = data['obs'], data['act'], data['adv'], data['logp']

        # Get the policy prediction, calculate loss using clip
        pi, logp = actor_ddp(obs, action)
        pi_ratio = torch.exp(logp - old_logp)
        clipped_adv = torch.clamp(pi_ratio, 1 - clip_ratio_offset, 1 + clip_ratio_offset) * advantage
        pi_loss = -(torch.min(pi_ratio * advantage, clipped_adv)).mean()
        return pi_loss
    
    def ppo_v_loss(data):
        obs, adj_rewards = data['obs'], data['rew']
        # Return the MSE loss from the value network
        return (critic_ddp(obs) - adj_rewards).pow(2).mean()
    
    def ppo_update():
        # Construct the minibatch buffer
        data = {k: torch.as_tensor(v, dtype=torch.float32)
                for k, v in dict(obs=obs_buf, act=act_buf, rew=rew_boot_buf, adv=adv_buf, logp=logp_buf).items()}

        # Train the policy network for a few iterations -- helps algo stability
        for i in range(train_pi_iters):
            pi_optim.zero_grad()
            pi_loss = ppo_pi_loss(data)
            pi_loss.backward()
            pi_optim.step()

        # Train the value network -- helps algo stability
        for i in range(train_v_iters):
            v_optim.zero_grad()
            v_loss = ppo_v_loss(data)
            v_loss.backward()
            v_optim.step()

    obs, ep_reward, ep_len = env.reset(), 0, 0
    ep_reward_history = [list() for _ in range(epochs)]
    for ep in range(epochs):
        for t in range(steps_per_epoch):
            # Get the predictions from the actor & critic networks
            action, val, logp_a = actor_critic(torch.as_tensor(obs, dtype=torch.float32))

            # Use the predicted action and step the environment
            new_obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_len += 1

            # Store the minibatch data for timestep t
            obs_buf[t] = obs
            act_buf[t] = action
            rew_buf[t] = reward
            val_buf[t] = val
            logp_buf[t] = logp_a

            # Update the old observation
            obs = new_obs

            # Check if the episode has ended
            timeout = ep_len == max_ep_len
            terminal = done or timeout
            epoch_ended = t == steps_per_epoch-1
            if terminal or epoch_ended:
                # Bootstrap the next episode using the current network state
                if timeout or epoch_ended:
                    _, bootstrap_v, _ = actor_critic(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    bootstrap_v = 0

                # Add the bootstrap value and reward to the minibatch buffer
                rews_aug = np.append(rew_buf[:], bootstrap_v)
                vals_aug = np.append(val_buf[:], bootstrap_v)
                
                # GAE-Lambda advantage calculation
                gae_deltas = rews_aug[:-1] + gamma * vals_aug[1:] - vals_aug[:-1]
                adv_buf[:] = discount_cumsum(gae_deltas, gamma * lam)
        
                # Computes rewards-to-go, to be targets for the value function
                rew_boot_buf[:] = discount_cumsum(rews_aug, gamma)[:-1]

                # Store the reward for the current epoch
                ep_reward_history[ep].append(ep_reward)
                obs, ep_reward, done = env.reset(), 0, 0

        # Update the networks using the data collected from the current minibatch
        ppo_update()

        # Average the minibatch rewards across all processes
        rank_rwd_mean = np.mean(ep_reward_history[ep])
        avg_rwd = ddp_avg_reward(rank, rank_rwd_mean)

        # Record the avg reward so we can save it to a file later
        if rank == 0:
            reward_rec.store(avg_rwd.item())
            print(f'device: {rank}, epoch: {ep + 1}, mean reward: {avg_rwd:.3f}')

    # Cleanup the distributed workers
    cleanup()

    # Dump the recorded values to a file
    if rank == 0:
        reward_rec.dump()

    # Return the trained model -- contains the value and policy networks
    return actor_critic


if __name__ == '__main__':
    mp.spawn(ppo, nprocs=num_workers, join=True)


