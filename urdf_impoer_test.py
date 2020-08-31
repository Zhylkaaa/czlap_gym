import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
planeId = p.loadURDF(fileName="plane.urdf")
boxId = p.loadURDF(fileName="../urdf_model/robot.urdf",basePosition=[0,0,0.2*np.sqrt(2)+0.03])

torque_tab = []

maxForce = 15
p.setJointMotorControl2(boxId, 3, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(-45))
p.setJointMotorControl2(boxId, 4, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(90))
p.setJointMotorControl2(boxId, 0, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(5))

p.setJointMotorControl2(boxId, 8, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(-45))
p.setJointMotorControl2(boxId, 9, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(90))
p.setJointMotorControl2(boxId, 5, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(-5))

p.setJointMotorControl2(boxId, 13, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(-45))
p.setJointMotorControl2(boxId, 14, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(90))
p.setJointMotorControl2(boxId, 10, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(5))

p.setJointMotorControl2(boxId, 18, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(-45))
p.setJointMotorControl2(boxId, 19, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(90))
p.setJointMotorControl2(boxId, 15, controlMode=p.POSITION_CONTROL, force=maxForce, targetPosition=np.deg2rad(-5))

# p.enableJointForceTorqueSensor(boxId, 4)

for i in range(p.getNumJoints(boxId)):
    print(p.getJointInfo(boxId, i), '\n')

for i in range (100000):
    p.stepSimulation()
    # torque_tab.append(p.getJointState(boxId, 4)[2][3])
    time.sleep(1./240.)

# print(torque_tab)
# plt.plot(torque_tab)
# plt.show()
p.disconnect()
