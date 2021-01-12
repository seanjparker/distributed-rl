import ray
import ray.rllib.agents.ppo as ray_ppo
from ray.tune.logger import pretty_print
from algos.common.util import EpochRecorder
from time import time


def ppo(current_workers, epochs):
    ray.init(num_cpus=current_workers)

    config = ray_ppo.DEFAULT_CONFIG.copy()
    config['num_gpus'] = 0
    config['num_workers'] = current_workers
    config['clip_param'] = 0.2
    config['num_sgd_iter'] = 30
    config['lambda'] = 0.97
    config['framework'] = 'tf2'
    config['model']['fcnet_hiddens'] = [64, 64]

    print(pretty_print(config))

    trainer = ray_ppo.PPOTrainer(config=config, env="LunarLander-v2")

    reward_rec = EpochRecorder(0, 'mean reward')

    start_time = time()
    for ep in range(epochs):
        result = trainer.train()
        avg_reward = result["episode_reward_mean"]
        reward_rec.store(avg_reward)
        print(f'epoch: {ep + 1}, mean reward: {avg_reward:.3f}')

    tot_time = time() - start_time
    reward_rec.dump(custom_data={
        'framework': 'ray',
        'd_lib': 'ray',
        'workers': current_workers,
        'time': str(tot_time)
    })
