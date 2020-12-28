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
#from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
from typing import Callable

if __name__ == '__main__':
    env_id = 'CzlapCzlap-v0'
    num_cpu = 1

    env_kwargs = dict(simulation_step=1 / 750., control_freq=1 / 30.)
    # env = DummyVecEnv([lambda: gym.make(env_id)])
    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv,
                       seed=0, env_kwargs=env_kwargs)

    env = VecFrameStack(env, n_stack=1)

    model = PPO.load('./checkpoints-2000-v12/rl_model_8377600_steps')
    #model = PPO.load('model_1000_baselines_v19')

    env = VecNormalize.load('./checkpoints-2000-v12/rl_env_8377600_steps', env)
    #env = VecNormalize.load('env_1000_baselines_v19', env)

    obs = env.reset()
    rews = []
    for _ in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        action_start = time.time()
        obs, reward, done, info = env.step(action)
        action_end = time.time()
        rews.append(reward)
        if action_end - action_start < 1/10:
            pass
            #time.sleep(1/10)
        if done:
            obs = env.reset()
            print(np.sum(rews))
            rews = []
