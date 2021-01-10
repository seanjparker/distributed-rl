import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import signal

from mpi4py import MPI
from algos.common.mpi import mpi_fork, mpi_proc_id, mpi_avg_scalar
from algos.common.util import EpochRecorder

# Hyperparameter definitions
num_workers = 4
pi_lr = 3e-4
v_lr = 1e-3
# epochs = 50
steps_per_epoch = 4000 // num_workers
max_ep_len = 1000
train_iters = 80
gamma = 0.99
lam = 0.97
clip_ratio_offset = 0.2


def tf_mpi_init():
    seed = 1337 * (mpi_proc_id() + 1)
    tf.random.set_seed(seed)
    np.random.seed(seed)


class MpiAdamOptimizer(tf.Module):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, var_list, lr):
        super().__init__()
        self.var_list = var_list
        self.lr = lr
        self.comm = MPI.COMM_WORLD
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-08
        self.t = tf.Variable(0, name='step', dtype=tf.int32)
        var_shapes = [v.shape.as_list() for v in var_list]
        self.var_sizes = [int(np.prod(s)) for s in var_shapes]
        self.flat_var_size = sum(self.var_sizes)
        self.m = tf.Variable(np.zeros(self.flat_var_size, 'float32'))
        self.v = tf.Variable(np.zeros(self.flat_var_size, 'float32'))

    def apply_gradients(self, flat_grad):
        buf = np.zeros(self.flat_var_size, np.float32)
        self.comm.Allreduce(flat_grad.numpy(), buf, op=MPI.SUM)
        avg_flat_grad = np.divide(buf, float(self.comm.Get_size()))
        self._apply_gradients(tf.constant(avg_flat_grad), lr=self.lr)
        if self.t.numpy() % 100 == 0:
            self.check_synced(tf.reduce_sum(self.var_list[0]).numpy())

    @tf.function
    def _apply_gradients(self, avg_flat_grad, lr):
        self.t.assign_add(1)
        t = tf.cast(self.t, tf.float32)
        a = lr * tf.math.sqrt(1 - tf.math.pow(self.beta2, t)) / (1 - tf.math.pow(self.beta1, t))
        self.m.assign(tf.multiply(self.beta1, self.m) + (1 - self.beta1) * avg_flat_grad)
        self.v.assign(tf.multiply(self.beta2, self.v) + (1 - self.beta2) * tf.math.square(avg_flat_grad))
        flat_step = tf.multiply(-a, self.m) / (tf.math.sqrt(self.v) + self.epsilon)
        var_steps = tf.split(flat_step, self.var_sizes, axis=0)
        for var_step, var in zip(var_steps, self.var_list):
            var.assign_add(tf.reshape(var_step, var.shape))

    def check_synced(self, localval):
        vals = self.comm.gather(localval)
        if self.comm.rank == 0:
            assert all(val == vals[0] for val in vals[1:]), f'Workers have different weights: {vals}'


class ActorCritic(tf.Module):
    def __init__(self, obs_space, act_space):
        super(ActorCritic, self).__init__(name='ppo')
        obs_dimensions = obs_space.shape[0]
        self.act_dim = act_space.n

        # policy network
        self.pi = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(obs_dimensions,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(act_space.n, activation='linear')
        ])

        # value network
        self.v = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(obs_dimensions,)),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(64, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        self.pi_optim = MpiAdamOptimizer(self.pi.trainable_variables, lr=pi_lr)
        self.v_optim = MpiAdamOptimizer(self.v.trainable_variables, lr=v_lr)

        values = MPI.COMM_WORLD.bcast([var.numpy() for var in self.variables])
        for (var, val) in zip(self.variables, values):
            var.assign(val)

    def get_pi_distribution(self, obs):
        return tfp.distributions.Categorical(logits=self.pi(obs))

    def train(self, data):
        pi_grads, pi_loss, v_grads, v_loss = self.get_gradients(data)
        self.pi_optim.apply_gradients(pi_grads)
        self.v_optim.apply_gradients(v_grads)

    @tf.function
    def get_gradients(self, data):
        obs, actions, adv, rewards, old_logp = data['obs'], data['act'], data['adv'], data['rew'], data['logp']

        with tf.GradientTape() as pi_tape:
            # fwd pass
            pi = self.get_pi_distribution(obs)
            logp_a = pi.log_prob(actions)

            # loss
            pi_ratio = tf.exp(logp_a - old_logp)
            clipped_adv = tf.where(pi_ratio > 0, 1 + clip_ratio_offset, 1 - clip_ratio_offset) * adv
            pi_loss = -(tf.reduce_mean(tf.minimum(pi_ratio * adv, clipped_adv)))

        pi_grads = pi_tape.gradient(pi_loss, self.pi.trainable_variables)
        pi_grads = tf.concat([tf.reshape(g, (-1,)) for g in pi_grads], axis=0)

        with tf.GradientTape() as v_tape:
            # fwd pass
            v = tf.squeeze(self.v(obs), axis=1)

            # loss
            v_loss = tf.reduce_mean(tf.pow(v - rewards, 2))

        v_grads = v_tape.gradient(v_loss, self.v.trainable_variables)
        v_grads = tf.concat([tf.reshape(g, (-1,)) for g in v_grads], axis=0)

        return pi_grads, pi_loss, v_grads, v_loss

    def step(self, obs):
        obs_batch = tf.expand_dims(obs, axis=0)
        pi = self.get_pi_distribution(obs_batch)
        action = pi.sample()
        logp_a = pi.log_prob(action)
        val = tf.squeeze(self.v(obs_batch), axis=0)

        return action.numpy()[0], val.numpy()[0], logp_a.numpy()[0]

    def get_action(self, obs):
        return self.step(obs)[0]


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def ppo(workers, epochs):
    rank = mpi_proc_id()
    tf_mpi_init()

    reward_rec = EpochRecorder(rank, 'mean reward')

    env = gym.make('LunarLander-v2')
    obs_space = env.observation_space
    act_space = env.action_space

    actor_critic = ActorCritic(obs_space, act_space)

    obs_buf = np.zeros((steps_per_epoch, obs_space.shape[0]), dtype=np.float32)
    act_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    adv_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    rew_boot_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    val_buf = np.zeros(steps_per_epoch, dtype=np.float32)
    logp_buf = np.zeros(steps_per_epoch, dtype=np.float32)

    def ppo_update():
        # TODO: Advantage normalisation trick
        data = {k: tf.convert_to_tensor(v, dtype=tf.float32)
                for k, v in dict(obs=obs_buf, act=act_buf, rew=rew_boot_buf, adv=adv_buf, logp=logp_buf).items()}

        for i in range(train_iters):
            actor_critic.train(data)

    obs, ep_reward, ep_len = env.reset(), 0, 0
    ep_reward_history = [list() for _ in range(epochs)]
    for ep in range(epochs):
        for t in range(steps_per_epoch):
            action, val, logp_a = actor_critic.step(tf.convert_to_tensor(obs, dtype=tf.float32))

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
                    _, bootstrap_v, _ = actor_critic.step(tf.convert_to_tensor(obs, dtype=tf.float32))
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
            reward_rec.store(avg_rwd)
            print(f'proc id: {rank}, epoch: {ep + 1}, mean reward: {avg_rwd:.3f}')

    # Training complete, dump the data to JSON
    if mpi_proc_id() == 0:
        reward_rec.dump(custom_data={'framework': 'tf', 'd_lib': 'mpi', 'workers': workers})


if __name__ == '__main__':
    mpi_fork(num_workers)
    ppo(num_workers)
