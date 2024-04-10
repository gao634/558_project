import pybullet as p
import numpy as np
import argparse
import prm
import time
import matplotlib.pyplot as plt
import math
import torch

def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

class Maze:
    def __init__(self, lidar, visuals=False, thresh=0.05):
        # number of lidar rays
        self.lidar = lidar
        self.obstacle_id = []
        self.rid = -1
        self.goal = (-10, -10)
        self.visuals = visuals
        self.goalmarker = -1
        self.thresh = thresh
    def getEnvTensor(self):
        env = torch.tensor(self.env.obs)
        return env.view(-1)
    def loadENV(self, path):
        env = prm.ENV(path)
        self.env = env
        # load outer walls
        for x in range(env.length + 1):
            id = p.loadURDF("assets/cube.urdf", [x - 0.5, env.width + 0.5, 0.5])
            self.obstacle_id.append(id)
            id = p.loadURDF("assets/cube.urdf", [x + 0.5, - 0.5, 0.5])
            self.obstacle_id.append(id)
        for y in range(env.width + 1):
            id = p.loadURDF("assets/cube.urdf", [env.length + 0.5, y + 0.5, 0.5])
            self.obstacle_id.append(id)
            id = p.loadURDF("assets/cube.urdf", [-0.5, y - 0.5, 0.5])
            self.obstacle_id.append(id)
        for x in range(env.length):
            for y in range(env.width):
                if env.obs[x][y]:
                    id = p.loadURDF("assets/cube.urdf", [x + 0.5, y + 0.5, 0.5])
                    self.obstacle_id.append(id)
        return self.obstacle_id
    def loadAgent(self, x, y, path='assets/turtlebot.urdf'):
        id = p.loadURDF(path, [x, y, 0])
        return id
    def getPos(self):
        pos, orn = p.getBasePositionAndOrientation(self.rid)
        x, y, z = pos
        rr, rp, ry = p.getEulerFromQuaternion(orn)
        return (x, y, z, rr, rp, ry)
    def setPos(self, x, y, dir=None):
        if dir is None:
            dir = np.random.uniform(0, 2*np.pi)
        orn = p.getQuaternionFromEuler([0, 0, dir])
        pos = [x, y, 0]
        p.resetBasePositionAndOrientation(self.rid, pos, orn)
        p.setJointMotorControl2(self.rid, 0, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.rid, 1, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    def setGoal(self, goal):
        self.goal = goal
        if self.visuals:
            self.goalmarker = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 0.5], 
                                       visualFramePosition=[goal[0], goal[1], 0.5])
    def collision(self):
        for obs in self.obstacle_id:
            contact = p.getContactPoints(bodyA=self.rid, bodyB=obs)
            if contact:
                return True
        return False
    def step(self, action=None):
        if action is None:
            return self.getInput(), 0, False, False
        t0, t1 = action
        p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=t0)
        p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=t1)
        p.stepSimulation()
        data = self.getInput()
        # hit wall or flipped
        collision = False
        # reached goal
        terminated = False
        x, y, z, rr, rp, ry = self.getPos()
        distance = dist((x,y), self.goal)
        if self.visuals:
            p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=ry*180/np.pi-90, cameraPitch=-60, cameraTargetPosition=(x, y, z))
        #    time.sleep(0.05)
        # robot leaves ground
        if z > 0.1 or z < -0.5:
            collision = True
        if abs(rr) > 10 or abs(rp) > 10:
            collision = True
        if self.collision():
            collision = True
        if distance < self.thresh:
            terminated = True
        if collision:
            reward = -10
        elif terminated:
            reward = 100
        else:
            reward = 1/distance - 0.1
        return data, reward, terminated, collision
    def getInput(self):
        x, y, z, rr, rp, ry = self.getPos()
        sensor_pos = (x, y, 0.9)
        num_rays = self.lidar
        max_range = 5
        angle = 2* np.pi / num_rays
        data = []
        for i in range(num_rays):
            dest = [x + max_range * np.cos(i * angle + ry), y + max_range * np.sin (i * angle + ry), 0.9]
            ray = p.rayTest(sensor_pos, dest)
            data.append(ray[0][2] * max_range if ray else max_range)
        return data
    def reset(self):
        if p.isConnected():
            p.disconnect()
        if self.visuals:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        id = p.loadURDF('assets/ground.urdf', [0, 0, -0.1])
        p.setGravity(0, 0, -9.81) 
        self.rid = self.loadAgent(-5, -5)
    def lidarGraph(self, data):
        n = len(data)
        angle = 2* np.pi / n
        x = [d * np.cos(i * angle) for i, d in enumerate(data)]
        y = [d * np.sin(i * angle) for i, d in enumerate(data)]
        for i in range(n):   
            plt.plot(x[i], y[i], 'o')
        plt.grid(True)
        plt.savefig('lidar.png')
        plt.show()