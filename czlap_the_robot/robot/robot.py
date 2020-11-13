import numpy as np
import yaml

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient
from typing import Tuple, Optional
from czlap_the_robot.robot.motor_controller import Controller

from simple_pid import PID

class Robot:
    """
    Wrapper on Czlap-Czlap PyBullet simulation

    Args:
        bullet_client (BulletClient): pybullet client object to interact with
        urdf_path (str): relative or absolute path to robot.urdf file defining robot structure
        props_path (str): relative or absolute path to props.yaml file defining props file
        xyz (tuple[float], optional): tuple containing x, y, z position to place robot at the start of simulation
            Defaults to (0, 0, 0)
        rpy (tuple[float], optional): tuple containing roll, pitch, yaw of robot as the start of simulation.
            Defaults to (0, 0, 0)
        control_mode (int, optional): pybullet predefined control method. Defaults to p.POSITION_CONTROL
        start_pos (np.array[float], optional): robot joints starting position. Defaults to None

    Attributes:
        _bc (BulletClient): pybullet client object to interact with
        _urdf_path (str): relative or absolute path to robot.urdf file defining robot structure
        _props_path (str): relative or absolute path to props.yaml file defining props file
        _start_xyz (tuple[float]): tuple containing x, y, z position to place robot at the start of simulation
        _start_rpy (tuple[float]): tuple containing roll, pitch, yaw of robot as the start of simulation
        _robot_id (int): robot id returned by loadURDF method of BulletClient
        _robot_data (dict): robot physical parameters stored in props.yaml ('coxa': hip, 'femur': thigh bone, 'tibia': shinbone)
        _joints_array (list): ids of robot's intractable joints
        _control_mode (int): pybullet predefined control method. Defaults to p.POSITION_CONTROL
        _start_joint_pos (np.array): robot's joints starting position
    """

    def __init__(
            self,
            bullet_client: BulletClient,
            urdf_path: str,
            props_path: str,
            xyz: Optional[Tuple[float, float, float]] = (0, 0, 0),
            rpy: Optional[Tuple[float, float, float]] = (0, 0, 0),
            control_mode: Optional[int] = p.POSITION_CONTROL,
            start_pos: Optional[np.array] = None
    ):
        self._bc = bullet_client
        self._urdf_path = urdf_path
        self._props_path = props_path
        self._start_xyz = xyz
        self._start_rpy = rpy
        self.joint_names = ['coxa', 'femur', 'tibia']
        self._robot_id = None
        self._robot_data = None
        self._link_len_arr = None
        self._coxa_link_len = None
        self._femur_link_len = None
        self._tibia_link_len = None

        self._joints_array = [2,  4,  6,
                              8,  10, 12,
                              14, 16, 18,
                              20, 22, 24]

        self._control_mode = control_mode
        self._start_joint_pos = None
        if start_pos is None:
            self._start_joint_pos = np.array([0.0, np.pi / 4, np.pi / 2] * 4)
        else:
            self._start_joint_pos = np.array(start_pos)

        self._spawn_robot()

    def _spawn_robot(self):
        """Place robot in simulation reset joint's position and velocity"""
        quaternion = self._bc.getQuaternionFromEuler(self._start_rpy)
        self._robot_id = self._bc.loadURDF(fileName=self._urdf_path,
                                           basePosition=self._start_xyz,
                                           baseOrientation=quaternion)
        self._import_props()
        self._reset_actuator_joints()
        self.reset_position()
        # self._set_mimic_joints()
        self.set_joint_array(self._start_joint_pos, np.zeros((12,)))

    def _import_props(self):
        """Read robot physical parameters"""
        with open(self._props_path, 'r') as stream:
            try:
                self._robot_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self._coxa_link_len = self._robot_data['coxa']['joint']['origin']['x']
        self._femur_link_len = self._robot_data['femur']['joint']['origin']['x']
        self._tibia_link_len = self._robot_data['tibia']['joint']['origin']['x']
        self._link_len_arr = np.array([self._coxa_link_len, self._femur_link_len, self._tibia_link_len])


    def _reset_actuator_joints(self):
        """Sets actuators be pure inertia without any dumping not spring stiffness"""
        for i in range(0, self._bc.getNumJoints(self._robot_id), 2):
            self._bc.setJointMotorControl2(self._robot_id, i,
                                           self._bc.VELOCITY_CONTROL,
                                           targetVelocity=0,
                                           force=0)
            self._bc.changeDynamics(self._robot_id, i, linearDamping=0, angularDamping=0)


    def _set_mimic_joints(self):
        """Have no fucking clue"""
        joint_num_map = {'coxa': (2, 1), 'femur': (4, 3), 'tibia': (6, 5)}
        for i in range(4):
            for joint in self.joint_names:
                c = self._bc.createConstraint(parentBodyUniqueId=self._robot_id,
                                              parentLinkIndex=joint_num_map[joint][0] + i * 6,
                                              childBodyUniqueId=self._robot_id,
                                              childLinkIndex=joint_num_map[joint][1] + i * 6,
                                              jointType=p.JOINT_GEAR,
                                              jointAxis=[1, 0, 0],
                                              parentFramePosition=[0, 0, 0],
                                              childFramePosition=[0, 0, 0])
                gear_ratio = 1 / self._robot_data[joint]['actuator']['gear_ratio']
                if joint == 'coxa' and i >= 2:
                    gear_ratio *= -1
                elif joint == 'femur':
                    if i % 2 == 0:
                        gear_ratio *= -1
                    if i >= 2:
                        gear_ratio *= -1
                elif joint == 'tibia' and i % 2 == 0:
                    gear_ratio *= -1
                self._bc.changeConstraint(c, gearRatio=gear_ratio, maxForce=10000)

    def _single_leg_inv_kinematicks(self, xyz):
        """
        Get single leg tip position relative to coxa base
        in [x, y, z] coordinates.

        Returns:
            np.array: [coxa_angle, femur_angle, tibia_angle]
            angles are in radians
        """

        assert np.linalg.norm(xyz) <= np.linalg.norm([self._link_len_arr[0], self._link_len_arr[1]+self._link_len_arr[2]])

        pow_xyz = np.power(xyz, 2)
        pow_link_len_arr = np.power(self._link_len_arr, 2)

        tibia_angle_num = np.sum(pow_xyz) - np.sum(pow_link_len_arr)
        tibia_angle_den = 2 * self._link_len_arr[1] * self._link_len_arr[2]

        # helping variables for future equations
        r = tibia_angle_num / tibia_angle_den       # cos(tibia_angle)
        q = 1 - r**2                                # sin(tibia_angle)**2

        tibia_angle = np.arccos(r) 
        

        femur_angle_den = (self._link_len_arr[1] + self._link_len_arr[2]*r)**2 + pow_link_len_arr[2]*q
        femur_angle_num = -xyz[0]*(self._link_len_arr[1]+r) - np.sqrt(pow_xyz[0]*(self._link_len_arr[1]+r)**2 + pow_link_len_arr[2]*q * femur_angle_den) 

        femur_angle = np.arccos(femur_angle_num/femur_angle_den)

        coxa_angle = np.pi/2 - np.arccos((self._link_len_arr[2]*(np.cos(tibia_angle)*np.sin(femur_angle) + np.cos(femur_angle)*np.sin(tibia_angle)) + self._link_len_arr[1]*np.sin(femur_angle)) / self._link_len_arr[0])
        return [coxa_angle, femur_angle-np.pi/2, tibia_angle]

    def get_body_pos_and_rpy(self):
        """
        Get body position and orientation from simulation
        position: [x, y, z] body coordinates
        orientation: [x, y, z] Euler angles (roll, pitch, yaw) in radians

        Returns:
            np.array: concatenate(position, orientation)

            Concatenated vector of position and orientation (total of 6 values)

        """
        pos, rpy = self._bc.getBasePositionAndOrientation(self._robot_id)
        rpy = self._bc.getEulerFromQuaternion(rpy)
        return np.concatenate((pos, rpy), axis=0).astype(np.float32)

    def get_body_velocity(self):
        """
        Get body linear and angular velocity from simulation

        Returns:
            np.array: concatenate(linear_velocity, angular_velocity)

            Concatenated vector of linear and angular velocity (total of 6 values)

        """
        linear_velocity, angular_velocity = self._bc.getBaseVelocity(self._robot_id)
        return np.concatenate((linear_velocity, angular_velocity), axis=0).astype(np.float32)

    def get_joint_pos_vel_array(self):
        """
        Get joints position and velocity from simulation

        Returns:
            np.array: concatenate(joint_positions, joint_velocities)

            Concatenated vector of joint positions and velocities (total of 24 values)

        """
        joint_states = self._bc.getJointStates(bodyUniqueId=self._robot_id,
                                               jointIndices=self._joints_array)

        joint_position = np.array(joint_states)[:, 0]  # TODO: something is wrong I can feel it...

        joint_velocity = np.array(joint_states)[:, 1]

        return np.concatenate((joint_position, joint_velocity), axis=0).astype(np.float32)

    def set_joint_array(self, target_positions, target_velocities=None):
        """
        Set joints next position

        Args:
            target_positions (np.array): target position for each joint (total of 12 radian angle values)
            target_velocities (np.array, optional): currently unused
        """
        assert np.shape(target_positions) == np.shape(self._joints_array)

        remapped_target_positions = np.copy(target_positions)
        remapped_target_positions[[1, 4]] = np.pi - target_positions[[1, 4]]
        remapped_target_positions[[2, 3, 5, 8, 9, 11]] = -target_positions[[2, 3, 5, 8, 9, 11]]

        remapped_target_velocities = None
        if target_velocities is None:
            remapped_target_velocities = self.max_vel_limits
        else:
            assert np.shape(target_velocities) == np.shape(self._joints_array)
            remapped_target_velocities = np.copy(target_velocities)
            remapped_target_velocities[[1, 2, 3, 4, 5, 8, 9, 11]] = -target_velocities[[1, 2, 3, 4, 5, 8, 9, 11]]

        for i in range(len(self._joints_array)):
            self._bc.setJointMotorControl2(bodyUniqueId=self._robot_id,
                                           jointIndex=self._joints_array[i],
                                           controlMode=self._control_mode,
                                           targetPosition=remapped_target_positions[i],
                                           maxVelocity=10,
                                           force=12)        # TODO Read thius stuff from URDF or YAML


    def reset_position(self, target_position=None):
        """
        Reset robot position, orientation, velocity and joint positions

        Args:
            target_position (np.array, optional): target position for each joint (total of 12 values). Defaults to self._start_joint_pos
        """
        if target_position is None:
            target_position = self._start_joint_pos

        helper_target_positnion = np.copy(target_position)
        helper_target_positnion[[1, 4]] = np.pi - target_position[[1, 4]]
        helper_target_positnion[[2, 3, 5, 8, 9, 11]] = -target_position[[2, 3, 5, 8, 9, 11]]

        remapped_target_position = np.zeros(np.shape(target_position)[0] * 2)
        remapped_target_position[np.array(self._joints_array)-1] = helper_target_positnion
        for i in range(1, self._bc.getNumJoints(self._robot_id)):
            self._bc.resetJointState(bodyUniqueId=self._robot_id,
                                     jointIndex=i,
                                     targetValue=remapped_target_position[i-1],
                                     targetVelocity=0)

        self._bc.resetBaseVelocity(self._robot_id,
                                   linearVelocity=0,
                                   angularVelocity=0)
        quaternion = self._bc.getQuaternionFromEuler(self._start_rpy)
        self._bc.resetBasePositionAndOrientation(bodyUniqueId=self._robot_id,
                                                 posObj=self._start_xyz,
                                                 ornObj=quaternion)

    @property
    def max_vel_limits(self):
        """Joint velocities limits (total of 12 values)"""
        max_vel_array = np.array([self._robot_data[joint]['actuator']['velocity'] /
                                  self._robot_data[joint]['actuator']['gear_ratio'] for joint in self.joint_names] * 4,
                                 dtype=np.float32)
        return max_vel_array

    @property
    def joint_pos_upper_limits(self):
        """Joint positions upper limits (total of 12 values)"""
        joints_limit_up = np.array(
            [self._robot_data[joint]['joint']['limit']['upper'] for joint in self.joint_names] * 4, dtype=np.float32)
        joints_limit_up = np.deg2rad(joints_limit_up)

        return joints_limit_up

    @property
    def joint_pos_lower_limits(self):
        joints_limit_low = np.array(
            [self._robot_data[joint]['joint']['limit']['lower'] for joint in self.joint_names] * 4, dtype=np.float32)
        joints_limit_low = np.deg2rad(joints_limit_low)

        return joints_limit_low
