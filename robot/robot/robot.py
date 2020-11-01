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

        self._joints_array = [1,  3,  5,
                              7,  9,  11,
                              13, 15, 17,
                              19, 21, 23]

        self._control_mode = control_mode
        self._start_joint_pos = None
        if start_pos == None:
            self._start_joint_pos = []
            for i in  range(4):
                self._start_joint_pos += (0.0, np.pi/4,  np.pi/2)
            self._start_joint_pos = np.array(self._start_joint_pos)
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
        self.setJointAngleArray(self._start_joint_pos)



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
        for i in range(0, 23, 6):
            c = self._bc.createConstraint(parentBodyUniqueId=self._robot_id,
                                          parentLinkIndex=1+i,
                                          childBodyUniqueId=self._robot_id,
                                          childLinkIndex=0+i,
                                          jointType=p.JOINT_GEAR,
                                          jointAxis=[1, 0, 0],
                                          parentFramePosition=[0, 0, 0],
                                          childFramePosition=[0, 0, 0])
            gearRatio = -1 / self._robot_data['coxa']['actuator']['gear_ratio']
            self._bc.changeConstraint(c, gearRatio=gearRatio, maxForce=10000)

            c = self._bc.createConstraint(parentBodyUniqueId=self._robot_id,
                                          parentLinkIndex=3+i,
                                          childBodyUniqueId=self._robot_id,
                                          childLinkIndex=2+i,
                                          jointType=p.JOINT_GEAR,
                                          jointAxis=[1, 0, 0],
                                          parentFramePosition=[0, 0, 0],
                                          childFramePosition=[0, 0, 0])
            gearRatio = -1 / self._robot_data['tibia']['actuator']['gear_ratio']
            self._bc.changeConstraint(c, gearRatio=gearRatio, maxForce=10000)

            c = self._bc.createConstraint(parentBodyUniqueId=self._robot_id,
                                          parentLinkIndex=5+i,
                                          childBodyUniqueId=self._robot_id,
                                          childLinkIndex=4+i,
                                          jointType=p.JOINT_GEAR,
                                          jointAxis=[1, 0, 0],
                                          parentFramePosition=[0, 0, 0],
                                          childFramePosition=[0, 0, 0])
            gearRatio = -1 / self._robot_data['femur']['actuator']['gear_ratio']
            self._bc.changeConstraint(c, gearRatio=gearRatio, maxForce=10000)



    def getBodyPosAndRPY(self):
        pos_and_rpy = self._bc.getBasePositionAndOrientation(self._robot_id)
        pos = pos_and_rpy[0]
        rpy = self._bc.getEulerFromQuaternion(pos_and_rpy[1])
        return np.concatenate((pos, rpy), axis=0)

    

    def getBodyVelocity(self):
        return self._bc.getBaseVelocity(self._robot_id)




    def getJointPosArray(self):
        # TODO first elements of array contain joint position
        joint_states = self._bc.getJointStates(bodyUniqueId=self._robot_id,
                                               jointIndices=self._joints_array)
        print(np.array(joint_states[0]))
        print('\n\n\n')
        return joint_states



    def setJointAngleArray(self, target_position):
        assert np.shape(target_position) == np.shape(self._joints_array)

        remapped_target_position = np.copy(target_position)
        remapped_target_position[[1, 4]] = np.pi-target_position[[1, 4]]
        remapped_target_position[[2, 3, 5, 8, 9, 11]] = -target_position[[2, 3, 5, 8, 9, 11]]

        self._bc.setJointMotorControlArray(bodyUniqueId=self._robot_id,
                                           jointIndices=self._joints_array,
                                           controlMode=self._control_mode,
                                           targetPositions=remapped_target_position)


    def resetPosition(self, target_position=None):

        if target_position == None:
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