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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

if __name__ == '__main__':
    env_id = 'CzlapCzlap-v0'
    num_cpu = 4

    #env = DummyVecEnv([lambda: gym.make(env_id)])
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0)

    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    n_steps = 2048
    steps_per_env = n_steps // num_cpu

    checkpoint_callback = CheckpointCallback(save_freq=n_steps*2, save_path='./checkpoints/', verbose=1)

    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log='./test_stable_baseline/',
                batch_size=128, n_steps=steps_per_env)

    model.learn(total_timesteps=n_steps*1000, callback=checkpoint_callback)

    model.save('model_1000_baselines_v2')

    """model = PPO.load('model_100_baselines')

    obs = env.reset()

    for _ in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()"""

