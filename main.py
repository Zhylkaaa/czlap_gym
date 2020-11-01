import pybullet as p
import time
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np

from robot.robot.robot import Robot

if __name__ == '__main__':
    client = bc.BulletClient(connection_mode=p.GUI)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    client.loadURDF("plane.urdf", useMaximalCoordinates=True)

    rob = Robot(client, 'urdf/robot.urdf', 'urdf/config/props.yaml', (0,0,0.2), (0,0,0))


    client.setRealTimeSimulation(1)
    client.setGravity(0,0,-9.81)

    while (1):
        time.sleep(1./240.)
        rob.getJointPosArray()
