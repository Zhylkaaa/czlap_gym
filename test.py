import pybullet as p
import time
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", 0, 0, 0)
robot = p.loadURDF("urdf/robot.urdf", [0, 0, 1])
for i in range(p.getNumJoints(robot)):
  print(p.getJointInfo(robot, i))
  p.setJointMotorControl2(robot, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

for i in range(0, 19, 6):
  c = p.createConstraint(robot,
                        1+i,
                        robot,
                        0+i,
                        jointType=p.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
  p.changeConstraint(c, gearRatio=-1/15, maxForce=10000)

  c = p.createConstraint(robot,
                        3+i,
                        robot,
                        2+i,
                        jointType=p.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
  p.changeConstraint(c, gearRatio=-1/15, maxForce=10000)

  c = p.createConstraint(robot,
                        5+i,
                        robot,
                        4+i,
                        jointType=p.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
  p.changeConstraint(c, gearRatio=-1/15, maxForce=10000)

  # p.setJointMotorControl2(robot, 1+i, p.POSITION_CONTROL, targetPosition=0)
  # p.setJointMotorControl2(robot, 3+i, p.POSITION_CONTROL, targetPosition=2.21)
  # p.setJointMotorControl2(robot, 5+i, p.POSITION_CONTROL, targetPosition=-1.26)

# p.setJointMotorControlArray(robot, [0, 2, 3, 6, 8, 9, 12, 14, 15, 18, 20, 21], controlMode=p.POSITION_CONTROL,
#                           targetPositions=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


p.setRealTimeSimulation(1)
p.setGravity(0,0,-9.81)
while (1):
  time.sleep(1./240.)
