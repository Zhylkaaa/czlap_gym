import pybullet as p
import time
import pybullet_data
import pybullet_utils.bullet_client as bc
import numpy as np
import os
from czlap_the_robot.robot import Robot

if __name__ == '__main__':

    simulation_step = 1. / 2000.

    client = bc.BulletClient(connection_mode=p.GUI)
    client.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = client.loadURDF("plane.urdf", useMaximalCoordinates=True)


    print(pybullet_data.getDataPath())
    # client.setPhysicsEngineParameter(enableConeFriction=0)
    client.setTimeStep(simulation_step)

    dirname = os.path.dirname(__file__)
    robot_path = os.path.join(dirname, 'czlap_the_robot/urdf/robot.urdf')
    props_path = os.path.join(dirname, 'czlap_the_robot/urdf/props.yaml')
    rob = Robot(client, robot_path, props_path, (0,0,0.16*2*np.sqrt(2)), (0,0,0))

    client.setGravity(0,0,-9.81)
    client.addUserDebugParameter('slider',0,100)
    client.setRealTimeSimulation(1)

    while (1):
        pass