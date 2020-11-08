import gym
import numpy as np
from czlap_the_robot.robot import Robot
import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data
import os

class CzlapCzlapEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._client = bc.BulletClient(connection_mode=p.GUI)
        self._client.setGravity(0, 0, -9.81)

        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._client.loadURDF("plane.urdf", useMaximalCoordinates=True)

        dirname = os.path.dirname(__file__)  # is it portable solution?
        robot_path = os.path.join(dirname, '../urdf/robot.urdf')
        props_path = os.path.join(dirname, '../urdf/config/props.yaml')
        self.robot = Robot(self._client, robot_path, props_path, (0., 0., 0.2), (0., 0., 0))

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        pass

    def close(self):
        pass
