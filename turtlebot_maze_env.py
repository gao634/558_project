import pybullet as p
import numpy as np
import argparse
import prm
import time
import lidar as lidar
import matplotlib.pyplot as plt

class Maze:
    def __init__(self, lidar):
        # number of lidar rays
        self.lidar = lidar
        self.obstacle_id = []
        self.rid = self.loadAgent(-5, -5)
    def loadENV(self, path):
        env = prm.ENV(path)
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
    def loadAgent(x, y, path='assets/turtlebot.urdf'):
        id = p.loadURDF(path, [x, y, 0])
        return id
    def getPos(self):
        pos, orn = p.getBasePositionAndOrientation(id)
        x, y, z = pos
        rr, rp, ry = p.getEulerFromQuaternion(orn)
        return (x, y, z, rr, rp, ry)
    def setPos(self, x, y, dir=None):
        if dir is None:
            dir = np.random.uniform(0, 2*np.pi)
        orn = p.getQuaternionFromEuler([0, 0, dir])
        pos = [x, y, 0]
        p.resetBasePositionAndOrientation(self.rid, pos, orn)
    def collision(self):
        pass
    def step(self, action):
        t0, t1 = action
        p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=t0)
        p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=t1)
        p.stepSimulation()
        data = lidar.getInput(self.rid, self.lidar)
        # hit wall or flipped
        collision = False
        # reached goal
        terminated = False
        x, y, z, rr, rp, ry = self.getPos()
        # robot leaves ground
        if z > 0.1 or z < -0.5:
            collision = True
        if abs(rr) > 10 or abs(rp) > 10:
            collision = True
    def getInput(self):
        x, y, z, rr, rp, ry = self.getPos()
        sensor_pos = (x, y, 0.9)
        num_rays = self.lidar
        max_range = 5
        angle = 2* np.pi / num_rays
        data = []
        for i in range(num_rays):
            dest = [x + max_range * np.cos(i * angle), y + max_range * np.sin (i * angle), 0.9]
            ray = p.rayTest(sensor_pos, dest)
            data.append(ray[0][2] * max_range if ray else max_range)
        return data
    def reset():
        p.disconnect()
        p.connect(p.GUI)
        p.loadURDF('assets/ground.urdf', [0, 0, -0.1])
        p.setGravity(0, 0, -9.81) 
    def lidarGraph(self, data):
        n = len(data)
        angle = 2* np.pi / n
        x = [d * np.cos(i * angle) for i, d in enumerate(data)]
        y = [d * np.sin(i * angle) for i, d in enumerate(data)]
        for i in range(n):   
            plt.plot(x[i], y[i], 'o')
        plt.grid(True)
        plt.show()