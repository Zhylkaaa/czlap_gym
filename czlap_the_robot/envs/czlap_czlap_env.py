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
        self.simulation_step = simulation_step
        self._client.setTimeStep(simulation_step)
        #self._client.setRealTimeSimulation(real_time)
        self.samples_per_control = int(control_freq / simulation_step)  # number of simulation steps to take before 1 control command
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._client.loadURDF("plane.urdf", useMaximalCoordinates=True)

        dirname = os.path.dirname(__file__)  # is it portable solution?
        robot_path = os.path.join(dirname, '../urdf/robot.urdf')
        props_path = os.path.join(dirname, '../urdf/props.yaml')

        self.start_rpy = np.array((0., 0., np.pi/2))
        self.initial_position = self.start_xyz = np.array((0., 0., 0.25))

        self.robot = Robot(self._client, robot_path, props_path, xyz=self.start_xyz, rpy=self.start_rpy)

        self.joint_pos_lower_limits = self.robot.joint_pos_lower_limits
        self.joint_pos_upper_limits = self.robot.joint_pos_upper_limits

        self.action_space = gym.spaces.box.Box(
            low=-np.ones_like(self.joint_pos_lower_limits),
            high=np.ones_like(self.joint_pos_upper_limits))

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
            low=np.concatenate([observation_lower_limits, self.action_space.low.astype(np.float32)]),
            high=np.concatenate([observation_upper_limits, self.action_space.high.astype(np.float32)]))

        self.last_position_and_rpy = self.robot.get_body_pos_and_rpy()
        self.last_action = np.zeros_like(self.action_space.low)

        self.np_random, _ = gym.utils.seeding.np_random()
        self.time_reward = 0.01  # TODO: make it time punishment?
        self.done = False
        self.turn_additional_punishment = True
        self.completion_counter = 0

    def get_observations(self):
        observations = np.concatenate([self.robot.get_joint_pos_vel_array(),
                                       self.robot.get_body_pos_and_rpy(),
                                       self.robot.get_body_velocity(),
                                       self.last_action], axis=0)
        return observations

    # TODO: meybe something more complicated?
    def calculate_reward(self, old_pos_and_rpy, new_pos_and_rpy):

        y_dist_traveled = np.float32(new_pos_and_rpy[1] - old_pos_and_rpy[1])
        if y_dist_traveled > 0:
            y_dist_traveled *= 2

        y_dist_from_origin = np.float32(new_pos_and_rpy[1] - self.initial_position[1])

        height_punishment = new_pos_and_rpy[2] - old_pos_and_rpy[2]
        if np.abs(1 - new_pos_and_rpy[2] / old_pos_and_rpy[2]) <= 0.03:
            height_punishment = 0

        x_deviation_punishment = -np.abs(new_pos_and_rpy[0] - self.initial_position[0])
        # return y_dist_traveled + y_dist_from_origin + self.time_reward

        roll_punishment, pitch_punishment, yaw_punishment = -np.abs(new_pos_and_rpy[3:] - self.start_rpy)

        if self.turn_additional_punishment:
            roll_weight = 0.01
            pitch_weight = 0.01
            yaw_weight = 0.02
            height_weight = 0.02
            x_weight = 0.01
            dist_from_origin_weight = 0
        else:
            roll_weight = 0.0005
            pitch_weight = 0.0005
            yaw_weight = 0.002  # 0.006
            height_weight = 0  #0.001
            x_weight = 0
            dist_from_origin_weight = 0.0005

        return y_dist_traveled + \
                   dist_from_origin_weight * y_dist_from_origin + \
                   height_weight * height_punishment + \
                   x_weight * x_deviation_punishment + \
                   roll_weight * roll_punishment + \
                   pitch_weight * pitch_punishment + \
                   yaw_weight * yaw_punishment

    def perform_simulation(self):
        for _ in range(self.samples_per_control):
            self._client.stepSimulation()

    def convert_actions(self, actions):
        """
        convert actions from [-1, 1] to target range, depending on joint limits
        actions: array of self.joint_pos_upper_limits.shape shape with values from [-1, 1] interval

        Returns:
             actions: array of self.joint_pos_upper_limits.shape shape with values from target interval (e.g. [0, pi]
        """

        joint_action_range = self.joint_pos_upper_limits - self.joint_pos_lower_limits
        actions = self.joint_pos_lower_limits + (joint_action_range / 2.) * (actions + 1)
        return actions

    def step(self, action: np.ndarray):
        if self.done:
            warnings.warn('making actions after episode is done may lead to unexpected behaviour')

        # TODO: punish for limit violation? or embed this in policy?
        # action = np.clip(action, self.robot.joint_pos_lower_limits, self.robot.joint_pos_upper_limits)

        # now actions are bound to [-1, 1] interval, so should be rescaled to limits
        self.last_action = np.copy(action)
        action = self.convert_actions(action)

        self.robot.set_joint_array(action)
        self.perform_simulation()

        obs = self.get_observations()

        position_and_rpy = obs[24:24+6]
        reward = self.calculate_reward(self.last_position_and_rpy, position_and_rpy)
        self.done = False

        self.last_position_and_rpy = position_and_rpy

        # robot has "fallen" if tilted front of backward more than 30 degree
        # or if rolled on side
        # or if has fallen
        # or if deviated from original x axis position mor than 2 meters
        if position_and_rpy[4] > np.pi/6 or position_and_rpy[4] < -np.pi/6 or \
                position_and_rpy[3] >= 0.9*np.pi/2 or position_and_rpy[3] <= -0.9*np.pi/2 \
                or position_and_rpy[2] <= 0.1 \
                or position_and_rpy[0] <= -2 or position_and_rpy[0] >= 2:
            self.done = True
            reward -= 1.
        elif position_and_rpy[1] >= 5.:
            self.completion_counter += 1
            if self.completion_counter >= 50:
                self.turn_additional_punishment = True
            reward += 1.
            self.done = True

        return obs, reward, self.done, dict()

    def reset(self):
        self.robot._reset_actuator_joints()
        self.robot.reset_position()
        self.robot.set_joint_array(self.robot._start_joint_pos)
        self.perform_simulation()
        self.last_position_and_rpy = self.robot.get_body_pos_and_rpy()

        """self.perform_simulation()
        new_pos_and_rpy = self.robot.get_body_pos_and_rpy()

        while not np.allclose(self.last_position_and_rpy[2], new_pos_and_rpy[2]):
            self.last_position_and_rpy = new_pos_and_rpy
            self.perform_simulation()
            new_pos_and_rpy = self.robot.get_body_pos_and_rpy()"""

        self.initial_position = self.last_position_and_rpy
        self.last_action = np.zeros_like(self.action_space.low)

        self.done = False

        return self.get_observations()

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        self._client.disconnect()
