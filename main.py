import pybullet as p
import time
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import gym
import czlap_the_robot
from policy import Policy
import yaml
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == '__main__':
    env_id = 'CzlapCzlap-v0'
    num_cpu = 2

    env_kwargs = dict(simulation_step=1/750)
    #env = DummyVecEnv([lambda: gym.make(env_id)])
    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv, seed=0, env_kwargs=env_kwargs)

    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecFrameStack(env, n_stack=1)

    n_steps = 2048
    steps_per_env = n_steps // num_cpu
    #steps_per_env = n_steps

    checkpoint_callback = CheckpointCallback(save_freq=n_steps*20, save_path='./checkpoints-1000-v227/', verbose=1)

    shared_net = [256]
    vf = [128]
    pi = [128]
    policy_kwargs = dict(net_arch=[*shared_net, dict(vf=vf, pi=pi)])
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log='./test_stable_baseline/',
                batch_size=64, n_steps=steps_per_env, n_epochs=15,
                learning_rate=linear_schedule(1e-4))

    model.learn(total_timesteps=n_steps*1000, callback=checkpoint_callback)

    model.save('model_1000_baselines_v227')

    """#model = PPO.load('./checkpoints-1000-v13/rl_model_819200_steps')
    model = PPO.load('model_1000_baselines_v13')

    obs = env.reset()
    rews = []
    for _ in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        action_start = time.time()
        obs, reward, done, info = env.step(action)
        action_end = time.time()
        rews.append(reward)
        if action_end - action_start < 1/10:
            time.sleep(1/10)
        if done:
            obs = env.reset()
            print(np.sum(rews))
            rews = []"""

