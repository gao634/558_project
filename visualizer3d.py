import pybullet as p
import numpy as np
import argparse
import prm
import time
import turtlebot_maze_env as maze

def movementDemo(id, env):
    keys = p.keys = p.getKeyboardEvents()
    vel = True
    for k, v in keys.items():
        if v & p.KEY_WAS_TRIGGERED:
            if k == p.B3G_UP_ARROW:
                #p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
                #p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
                _, t, c = env.step(0, vel=vel, discr=True)
            elif k == p.B3G_DOWN_ARROW:
                _, t, c = env.step(3, vel=vel, discr=True)
            elif k == p.B3G_LEFT_ARROW:
                _, t, c = env.step(2, vel=vel, discr=True)
            elif k == p.B3G_RIGHT_ARROW:
                _, t, c = env.step(1, vel=vel, discr=True)
    pos, orn = p.getBasePositionAndOrientation(id)
    x, y, z = pos
    rr, rp, ry = p.getEulerFromQuaternion(orn)
    #p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=ry*180/np.pi-90, cameraPitch=-60, cameraTargetPosition=pos)
    joint_state = p.getJointState(bodyUniqueId=id, jointIndex=0)
    applied_force = joint_state[3]
    #print("Applied force:", applied_force)
    #print(z, rr, rp)
    #print(env.goalAngle()[0])
    #print(p.getJointState(id, 0), p.getJointState(id, 1))
    print(env.goalWorldAngle(x, y), ry)
    p.stepSimulation()


def main(args):
    env = maze.Maze(180, visuals=True)
    env.reset()
    env.loadENV('data/envs/env_0.txt')
    x = 0.5
    y = 0.5
    env.setPos(x, y, env.goalWorldAngle(x, y), False)
    env.setGoal((2.5, 0.5))
    #env.lidarGraph(env.getFivePoint())
    #id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=2, rgbaColor=[1, 0, 0, 1], visualFramePosition=[0, 0, 3])
    while True:
        #movementDemo(id)
        movementDemo(env.rid, env)
        # Sleep to avoid consuming too much CPU
        time.sleep(0.01)

main(args=None)