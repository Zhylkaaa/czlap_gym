import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)
planeId = p.loadURDF(fileName="plane.urdf")
boxId = p.loadURDF(fileName="../urdf_model/robot.urdf",basePosition=[0,0,1])

# for i in range(p.getNumJoints(boxId)):
#     print(p.getJointInfo(boxId, i))

for _ in range (5000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()
