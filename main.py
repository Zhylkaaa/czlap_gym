import pybullet as p
import time
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from agent import TRPOAgent
import gym
import czlap_the_robot
from policy import Policy
import yaml

if __name__ == '__main__':

    # TODO(1): refactor this block
    with open('czlap_the_robot/urdf/config/props.yaml', 'r') as stream:
        robot_config = yaml.safe_load(stream)
    joint_names = ['coxa', 'femur', 'tibia']
    joints_limit_up = np.array(
        [robot_config[joint]['joint']['limit']['upper'] for joint in joint_names] * 4, dtype=np.float32)
    joints_limit_up = np.deg2rad(joints_limit_up)

    joints_limit_low = np.array(
        [robot_config[joint]['joint']['limit']['lower'] for joint in joint_names] * 4, dtype=np.float32)
    joints_limit_low = np.deg2rad(joints_limit_low)
    # TODO(1): end of block

    constraints = np.maximum(np.abs(joints_limit_low), np.abs(joints_limit_up))

    nn = Policy(36, 256, 12, constraints)

    agent = TRPOAgent(policy=nn)

    agent.save_model("agent.pth")
    agent.load_model("agent.pth")
    agent.train('CzlapCzlap-v0', seed=0, batch_size=5000, iterations=10000,
                max_episode_length=500, verbose=True)
    agent.save_model("agent.pth")

    """env = gym.make('CzlapCzlap-v0')
    ob = env.reset()
    while True:
        action = agent(ob)
        ob, _, done, _ = env.step(action)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1 / 30)"""
