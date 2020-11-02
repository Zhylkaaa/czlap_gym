import pybullet as p
import time
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import matplotlib.pyplot as plt

from robot.robot.robot import Robot

if __name__ == '__main__':
    client = bc.BulletClient(connection_mode=p.GUI)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane.urdf", useMaximalCoordinates=True)

    rob = Robot(client, 'urdf/robot.urdf', 'urdf/config/props.yaml', (0,0,0.2), (0,0,0))

    client.setGravity(0,0,-9.81)

    T = np.arange(0, 2*np.pi, 1./15)
    joint_move = np.sin(T) * np.pi/4
    print(np.shape(joint_move))

    i = 0
    while (1):
        client.stepSimulation()
        time.sleep(1./240.)
        joint_pos = np.array([joint_move[i], np.pi/4+joint_move[i],  np.pi/2+joint_move[i]]).reshape(3, 1)
        joint_pos = (np.ones((4, 1)) * joint_pos.T).reshape(1, 12)[0]
        rob.setJointArray(joint_pos)
        i += 1
        if i >= np.shape(joint_move)[0]-1:
            i = 0