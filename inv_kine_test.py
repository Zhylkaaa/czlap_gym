import pybullet as p
import time
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import os
from czlap_the_robot.robot import Robot

if __name__ == '__main__':
    client = bc.BulletClient(connection_mode=p.GUI)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane.urdf", useMaximalCoordinates=True)

    dirname = os.path.dirname(__file__)
    robot_path = os.path.join(dirname, 'czlap_the_robot/urdf/robot.urdf')
    props_path = os.path.join(dirname, 'czlap_the_robot/urdf/config/props.yaml')
    rob = Robot(client, robot_path, props_path, (0,0,1), (0,0,0))

    client.setGravity(0,0,-9.81)

    T = np.arange(0, 2*np.pi, 1./15)
    joint_move = np.sin(T) * np.pi/4

    i = 0
    while (1):
        client.stepSimulation()
        time.sleep(1./240.)
        # joint_pos = np.array([joint_move[i], np.pi/4+joint_move[i],  np.pi/2+joint_move[i]]).reshape(3, 1)
        # joint_pos = (np.ones((4, 1)) * joint_pos.T).reshape(1, 12)[0]
        
        leg_ang = rob._single_leg_inv_kinematicks([0, 0.15*np.sqrt(2)+0.01, 0.105])
        print(np.rad2deg(leg_ang))

        rob.set_joint_array(np.array(leg_ang * 4))
        # rob.set_joint_array(np.array([0, 0, 0] * 4))
        i += 1
        if i >= np.shape(joint_move)[0]-1:
            i = 0
