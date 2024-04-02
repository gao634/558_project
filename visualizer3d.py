import pybullet as p
import numpy as np
import argparse
import prm
import time

def loadENV(path):
    env = prm.ENV(path)
    p.resetDebugVisualizerCamera(cameraDistance=1.3 * max(env.length, env.width), cameraYaw=45, cameraPitch=-89.9, cameraTargetPosition=[env.length/2, env.width/2, 0])
    # load outer walls
    for x in range(env.length + 1):
        p.loadURDF("assets/cube.urdf", [x - 0.5, env.width + 0.5, 0.5])
        p.loadURDF("assets/cube.urdf", [x + 0.5, - 0.5, 0.5])
    for y in range(env.width + 1):
        p.loadURDF("assets/cube.urdf", [env.length + 0.5, y + 0.5, 0.5])
        p.loadURDF("assets/cube.urdf", [-0.5, y - 0.5, 0.5])
    for x in range(env.length):
        for y in range(env.width):
            if env.obs[x][y]:
                p.loadURDF("assets/cube.urdf", [x + 0.5, y + 0.5, 0.5])

# we only have 1 agent, our turtlebot
def loadAgent(x, y, path='assets/turtlebot.urdf'):
    p.loadURDF(path, [x, y, 0])


def main(args):
    p.connect(p.GUI)
    loadENV('env_0.txt')
    loadAgent(0.5, 0.5)
    p.loadURDF('assets/cylinder.urdf', [1.5, 0.5, 0.5])
    while True:
        camera_params = p.getDebugVisualizerCamera()
        dist = camera_params[10]
        yaw = camera_params[8]
        pitch = camera_params[9]
        x = camera_params[11][0]
        y = camera_params[11][1]
        keys = p.getKeyboardEvents()
        
        # Adjust camera parameters based on key presses
        for k, v in keys.items():
            if v & p.KEY_WAS_TRIGGERED:
                if k == p.B3G_UP_ARROW:
                    y += 0.3
                elif k == p.B3G_DOWN_ARROW:
                    y -= 0.3
                elif k == p.B3G_LEFT_ARROW:
                    x -= 0.3
                elif k == p.B3G_RIGHT_ARROW:
                    x += 0.3
        
        # Set camera position and orientation
        p.resetDebugVisualizerCamera(dist, yaw, pitch, [x, y, 0])
        
        # Sleep to avoid consuming too much CPU
        time.sleep(0.1)
main(args=None)