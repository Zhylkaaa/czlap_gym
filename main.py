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


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` steps

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param model_name_prefix: Common prefix to the saved models
    :param env_name_prefix: Common prefix to the saved envs
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, model_name_prefix: str = "rl_model",
                 env_name_prefix: str = "rl_env", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name_prefix = model_name_prefix
        self.env_name_prefix = env_name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.model_name_prefix}_{self.num_timesteps}_steps")
            env_path = os.path.join(self.save_path, f"{self.env_name_prefix}_{self.num_timesteps}_steps")
            self.model.save(model_path)
            self.model.get_env().save(env_path)

            if self.verbose > 1:
                print(f"Saving model checkpoint to {model_path}")

        return True


if __name__ == '__main__':
    env_id = 'CzlapCzlap-v0'
    num_cpu = 4

    env_kwargs = dict(simulation_step=1/750., control_freq=1/30.)
    #env = DummyVecEnv([lambda: gym.make(env_id)])
    env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv,
                       seed=0, env_kwargs=env_kwargs)

    env = VecFrameStack(env, n_stack=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    steps_per_env = 1540
    n_steps = steps_per_env * num_cpu

    #steps_per_env = n_steps

    checkpoint_callback = CheckpointCallback(save_freq=n_steps*20, save_path='./checkpoints-1000-v17/', verbose=2)

    shared_net = [128]
    vf = [128]
    pi = [128, 128]
    policy_kwargs = dict(net_arch=[*shared_net, dict(vf=vf, pi=pi)])
    model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log='./test_stable_baseline/',
                batch_size=128, n_steps=steps_per_env, n_epochs=15,
                learning_rate=linear_schedule(1e-4))

    model.learn(total_timesteps=n_steps*1000, callback=checkpoint_callback)
    
    model.save('model_1000_baselines_v17')
    env.save('env_1000_baselines_v17')

