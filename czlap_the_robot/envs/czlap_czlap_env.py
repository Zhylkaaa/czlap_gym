import gym
import numpy as np
from czlap_the_robot.robot import Robot
import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data
import os
import warnings
import time


class CzlapCzlapEnv(gym.Env):
    """
    observations: concatenate(joint_positions, joint_velocities, body_position, body_rpy,
                              body_linear_velocity, body_angular_velocity) (total of 36 values)

    actions: target_positions (angles in radians) for each joint (total of 12 values)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, simulation_step=1./2000., control_freq=1/10., real_time=0):
        self._client = bc.BulletClient(connection_mode=p.DIRECT)
        self._client.setGravity(0, 0, -9.81)
        self._client.setTimeStep(simulation_step)
        self._client.setRealTimeSimulation(real_time)
        self.samples_per_control = int(control_freq / simulation_step)  # number of simulation steps to take before 1 control command
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._client.loadURDF("plane.urdf", useMaximalCoordinates=True)

        dirname = os.path.dirname(__file__)  # is it portable solution?
        robot_path = os.path.join(dirname, '../urdf/robot.urdf')
        props_path = os.path.join(dirname, '../urdf/config/props.yaml')
        self.start_xyz = (0., 0., 0.2)
        self.start_rpy = (0., 0., np.pi/2)
        self.robot = Robot(self._client, robot_path, props_path, self.start_xyz, self.start_rpy)

        self.action_space = gym.spaces.box.Box(
            low=self.robot.joint_pos_lower_limits,
            high=self.robot.joint_pos_upper_limits)

        body_position_lower_limits = np.array([-np.inf, -np.inf, 0], dtype=np.float32)
        body_position_upper_limits = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        body_rpy_limits = np.array([np.pi, np.pi, np.pi], dtype=np.float32)
        body_linear_velocity_limits = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        body_angular_velocity_limits = np.array([np.inf, np.inf, np.inf], dtype=np.float32)

        observation_lower_limits = np.concatenate([self.robot.joint_pos_lower_limits,
                                                   -self.robot.max_vel_limits,
                                                   body_position_lower_limits,
                                                   -body_rpy_limits,
                                                   -body_linear_velocity_limits,
                                                   -body_angular_velocity_limits], axis=0)

        observation_upper_limits = np.concatenate([self.robot.joint_pos_upper_limits,
                                                   self.robot.max_vel_limits,
                                                   body_position_upper_limits,
                                                   body_rpy_limits,
                                                   body_linear_velocity_limits,
                                                   body_angular_velocity_limits], axis=0)

        self.observation_space = gym.spaces.box.Box(
            low=observation_lower_limits,
            high=observation_upper_limits)

        self.last_position_and_rpy = self.robot.get_body_pos_and_rpy()

        self.np_random, _ = gym.utils.seeding.np_random()
        self.time_reward = 0.01
        self.done = False

    def get_observations(self):
        observations = np.concatenate([self.robot.get_joint_pos_vel_array(),
                                       self.robot.get_body_pos_and_rpy(),
                                       self.robot.get_body_velocity()], axis=0)
        return observations

    # TODO: meybe something more complicated?
    def calculate_reward(self, old_position, new_position):
        y_dist_traveled = np.float32(new_position[1] - old_position[1])
        if y_dist_traveled > 0:
            y_dist_traveled *= 2

        y_dist_from_origin = 0.5 * np.float32(new_position[1] - self.start_xyz[1])

        return y_dist_traveled + y_dist_from_origin + self.time_reward

    def perform_simulation(self):
        for _ in range(self.samples_per_control):
            self._client.stepSimulation()

    def step(self, action):
        if self.done:
            warnings.warn('making actions after episode is done may lead to unexpected behaviour')

        # TODO: punish for limit violation? or embed this in policy?
        action = np.clip(action, self.robot.joint_pos_lower_limits, self.robot.joint_pos_upper_limits)

        self.robot.set_joint_array(action)
        self.perform_simulation()

        obs = self.get_observations()

        position, rpy = obs[24:24+3], obs[27:27+3]
        last_position, last_rpy = self.last_position_and_rpy[:3], self.last_position_and_rpy[3:]
        reward = self.calculate_reward(last_position, position)
        self.done = False

        self.last_position_and_rpy = np.concatenate((position, rpy), axis=0)

        # robot has "fallen" if tilted front of backward more than 30 degree or if rolled on side or if has fallen
        if rpy[1] > np.pi/6 or rpy[1] < -np.pi/6 or rpy[0] >= np.pi/2 or rpy[0] <= -np.pi/2 or position[2] <= 0.1:
            self.done = True
            # reward -= 3.
        elif position[1] >= 10.:
            reward += 50.
            self.done = True

        return obs, reward, self.done, dict()

    def reset(self):
        self.robot.reset_position()
        self.last_position_and_rpy = self.robot.get_body_pos_and_rpy()
        self.done = False
        self.perform_simulation()
        return self.get_observations()

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        self._client.disconnect()
