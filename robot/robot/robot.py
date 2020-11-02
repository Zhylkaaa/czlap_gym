import numpy as np
import yaml

import pybullet as p
import pybullet_data

class Robot:
    def __init__(self, bullet_client, urdf_path, props_path, xyz=(0,0,0), rpy=(0,0,0), control_mode=p.POSITION_CONTROL, start_pos=None):
        self._bc = bullet_client
        self._urdf_path = urdf_path
        self._props_path = props_path
        self._start_xyz = xyz
        self._start_rpy = rpy
        self._robot_id = None
        self._robot_data = None
        self._position_gains = [0.1 for _ in range(12)]
        self._velocity_gains = [1.4 for _ in range(12)]

        self._joints_array = [1,  3,  5,
                              7,  9,  11,
                              13, 15, 17,
                              19, 21, 23]

        self._control_mode = control_mode
        self._start_joint_pos = None
        if start_pos is None:
            self._start_joint_pos = np.array([0.0, np.pi/4,  np.pi/2]).reshape(3, 1)
            self._start_joint_pos = (np.ones((4, 1)) * self._start_joint_pos.T).reshape(1, 12)[0]
        else:
            self._start_joint_pos = np.array(start_pos)

        self._spawnRobot()



    def _spawnRobot(self):
        quaternion = self._bc.getQuaternionFromEuler(self._start_rpy)
        self._robot_id = self._bc.loadURDF(fileName=self._urdf_path,
                                           basePosition=self._start_xyz,
                                           baseOrientation=quaternion)
        self._importProps()
        self._resetActuatorJoints()
        self.resetPosition()
        self._setMimicJoints()
        self.setJointArray(self._start_joint_pos, np.array([0 for _ in range(12)]))



    def _importProps(self):
        with open(self._props_path, 'r') as stream:
            try:
                self._robot_data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)



    def _resetActuatorJoints(self):
        for i in range(0, self._bc.getNumJoints(self._robot_id), 2):
            self._bc.setJointMotorControl2(self._robot_id, i,
                                           self._bc.VELOCITY_CONTROL,
                                           targetVelocity=0,
                                           force=0)



    def _setMimicJoints(self):
        joint_num_map = {'coxa':(1,0), 'femur':(3,2), 'tibia':(5,4)}
        for i in range(0, 23, 6):
            for joint in ('coxa', 'femur', 'tibia'):
                c = self._bc.createConstraint(parentBodyUniqueId=self._robot_id,
                                            parentLinkIndex=joint_num_map[joint][0]+i,
                                            childBodyUniqueId=self._robot_id,
                                            childLinkIndex=joint_num_map[joint][1]+i,
                                            jointType=p.JOINT_GEAR,
                                            jointAxis=[1, 0, 0],
                                            parentFramePosition=[0, 0, 0],
                                            childFramePosition=[0, 0, 0])
                gearRatio = -1 / self._robot_data[joint]['actuator']['gear_ratio']
                self._bc.changeConstraint(c, gearRatio=gearRatio, maxForce=10000)



    def getBodyPosAndRPY(self):
        pos_and_rpy = self._bc.getBasePositionAndOrientation(self._robot_id)
        pos = pos_and_rpy[0]
        rpy = self._bc.getEulerFromQuaternion(pos_and_rpy[1])
        return np.concatenate((pos, rpy), axis=0)

    

    def getBodyVelocity(self):
        return self._bc.getBaseVelocity(self._robot_id)




    def getJointPosVelArray(self):
        joint_states = self._bc.getJointStates(bodyUniqueId=self._robot_id,
                                               jointIndices=self._joints_array)

        joint_position = np.array(joint_states)[:,0]
        joint_position[[1, 4]] = np.pi-joint_position[[1, 4]]
        joint_position[[2, 3, 5, 8, 9, 11]] = -joint_position[[2, 3, 5, 8, 9, 11]]

        joint_velocity = np.array(joint_states)[:,1]
        joint_velocity[[1, 2, 3, 4, 5, 8, 9, 11]] = -joint_velocity[[1, 2, 3, 4, 5, 8, 9, 11]]

        return np.concatenate((joint_position, joint_velocity), axis=0)



    def setJointArray(self, target_positions, target_velocities=None):
        assert np.shape(target_positions) == np.shape(self._joints_array)

        remapped_target_positions = np.copy(target_positions)
        remapped_target_positions[[1, 4]] = np.pi-target_positions[[1, 4]]
        remapped_target_positions[[2, 3, 5, 8, 9, 11]] = -target_positions[[2, 3, 5, 8, 9, 11]]


        remapped_target_velocities = None
        if target_velocities is None:
            remapped_target_velocities = self.max_vel_limits
        else:
            assert np.shape(target_velocities) == np.shape(self._joints_array)
            remapped_target_velocities = np.copy(target_velocities)
            remapped_target_velocities[[1, 2, 3, 4, 5, 8, 9, 11]] = -target_velocities[[1, 2, 3, 4, 5, 8, 9, 11]]


        self._bc.setJointMotorControlArray(bodyUniqueId=self._robot_id,
                                           jointIndices=self._joints_array,
                                           controlMode=self._control_mode,
                                           targetPositions=remapped_target_positions,
                                           targetVelocities=remapped_target_velocities,
                                           positionGains=self._position_gains,
                                           velocityGains=self._velocity_gains)


    def resetPosition(self, target_position=None):

        if target_position is None:
            target_position = self._start_joint_pos

        helper_target_positnion = np.copy(target_position)
        helper_target_positnion[[1, 4]] = np.pi-target_position[[1, 4]]
        helper_target_positnion[[2, 3, 5, 8, 9, 11]] = -target_position[[2, 3, 5, 8, 9, 11]]

        remapped_target_position = np.zeros(np.shape(target_position)[0]*2)
        remapped_target_position[self._joints_array] = helper_target_positnion
        for i in range(0, self._bc.getNumJoints(self._robot_id)):
            self._bc.resetJointState(bodyUniqueId=self._robot_id,
                                     jointIndex=i,
                                     targetValue=remapped_target_position[i],
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
        joint_names = ['coxa', 'femur', 'tibia']
        max_vel_array = np.array([self._robot_data[joint]['actuator']['velocity'] /
                                  self._robot_data[joint]['actuator']['gear_ratio'] for joint in joint_names]).reshape(3,1)
        return (np.ones((4, 1)) * max_vel_array.T).reshape(1, 12)[0]

    @property
    def joint_pos_limits(self):
        joint_names = ['coxa', 'femur', 'tibia']
        joitnt_limit_up = np.array([self._robot_data[joint]['joint']['limit']['upper'] for joint in joint_names]).reshape(3,1)
        joitnt_limit_up = np.deg2rad(joitnt_limit_up)

        joitnt_limit_low = np.array([self._robot_data[joint]['joint']['limit']['lower'] for joint in joint_names]).reshape(3,1)
        joitnt_limit_low = np.deg2rad(joitnt_limit_low)

        return np.array([(np.ones((4, 1)) * joitnt_limit_low).T.reshape(1, 12)[0],
                         (np.ones((4, 1)) * joitnt_limit_up).T.reshape(1, 12)[0]])
