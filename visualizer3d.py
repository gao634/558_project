import pybullet as p
import numpy as np
import argparse
import prm
import time
import lidar as lidar
import turtlebot_maze_env as maze

def velocityDemo(id, vel):
    keys = p.keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if v & p.KEY_WAS_TRIGGERED:
            if k == p.B3G_UP_ARROW:
                vel += 10
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
                p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
            elif k == p.B3G_DOWN_ARROW:
                vel -= 10
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
                p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
            elif k == p.B3G_LEFT_ARROW:
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=-vel, force=100)
                p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
            elif k == p.B3G_RIGHT_ARROW:
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
                p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=-vel, force=100)
    pos, orn = p.getBasePositionAndOrientation(id)
    x, y, z = pos
    rr, rp, ry = p.getEulerFromQuaternion(orn)
    p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=ry*180/np.pi-90, cameraPitch=-60, cameraTargetPosition=pos)
    joint_state = p.getJointState(bodyUniqueId=id, jointIndex=0)
    applied_force = joint_state[3]
    #print("Applied force:", applied_force)
    #print(z, rr, rp)
    p.stepSimulation()

    return vel

def movementDemo(id):
    k = p.getKeyboardEvents()
    is_button_pressed = False
    # Check if the up arrow key is pressed
    if k == p.B3G_UP_ARROW:
        print('up')
        # Set the button state to pressed
        is_button_pressed = True
    elif k == p.B3G_UP_ARROW + p.KEY_WAS_RELEASED:
        # Set the button state to released
        is_button_pressed = False

    # Apply torque if the button is pressed
    if is_button_pressed:
        print('Moving forward')
        # Apply equal forces to both wheels to move forward
        force = 100  # Adjust the force as needed
        p.setJointMotorControl2(id, 0, p.TORQUE_CONTROL, force=force)
        p.setJointMotorControl2(id, 1, p.TORQUE_CONTROL, force=force)
    else:
        # Stop applying torque if the button is released
        p.setJointMotorControl2(id, 0, p.TORQUE_CONTROL, force=0)
        p.setJointMotorControl2(id, 1, p.TORQUE_CONTROL, force=0)
    for k, v in keys.items():
        if v & p.KEY_WAS_TRIGGERED:
            if k == p.B3G_UP_ARROW:
                p.setJointMotorControl2(id, 0, p.TORQUE_CONTROL, force=force)
                p.setJointMotorControl2(id, 1, p.TORQUE_CONTROL, force=force)
            elif k == p.B3G_DOWN_ARROW:
                print('down')
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=-1)
                p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            elif k == p.B3G_LEFT_ARROW:
                print('left')
                # left joint is 0, right joint is 1
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=-1)
                #p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
            elif k == p.B3G_RIGHT_ARROW:
                print('right')
                p.setJointMotorControl2(id, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
                #p.setJointMotorControl2(id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=-1)
    pos, orn = p.getBasePositionAndOrientation(id)
    x, y, z = pos
    rr, rp, ry = p.getEulerFromQuaternion(orn)
    p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=ry-90, cameraPitch=-60, cameraTargetPosition=pos)
    p.stepSimulation()

def main(args):
    env = maze.Maze(180, visuals=True)
    env.reset()
    env.loadENV('data/envs/env_0.txt')
    env.setPos(0.5, 0.5, 0)
    vel = 0
    while True:
        #movementDemo(id)
        vel = velocityDemo(env.rid, vel)
        # Sleep to avoid consuming too much CPU
        time.sleep(0.01)

main(args=None)