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
        self.steps = 0
        self.prevObs = None
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
    def getObs(self):
        pos = self.getPos()
        vel = self.getVel()
        input = self.getFivePoint()
        goal = self.goalAngle()
        return pos, vel, input, goal
    def getPos(self):
        pos, orn = p.getBasePositionAndOrientation(self.rid)
        x, y, z = pos
        rr, rp, ry = p.getEulerFromQuaternion(orn)
        return (x, y, z, rr, rp, ry)
    def getVel(self):
        vel0 = p.getJointState(self.rid, 0)[1]
        vel1 = p.getJointState(self.rid, 1)[1]
        return (vel0, vel1)
    def setPos(self, x, y, dir=None, rnd=True):
        if dir is None:
            dir = np.random.uniform(0, 2*np.pi)
        elif rnd:
            dir = np.random.normal(dir, 1)
        orn = p.getQuaternionFromEuler([0, 0, dir])
        pos = [x, y, 0]
        p.resetBasePositionAndOrientation(self.rid, pos, orn)
        p.setJointMotorControl2(self.rid, 0, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.rid, 1, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.resetDebugVisualizerCamera(cameraDistance=7, cameraYaw=90, cameraPitch=-75, cameraTargetPosition=(x, y, 0))
        self.steps = 0
    def setGoal(self, goal):
        if self.goalmarker > -1:
            p.removeBody(self.goalmarker)
        self.goal = goal
        if self.visuals:
            self.goalmarker = p.loadURDF('assets/marker.urdf', [goal[0], goal[1], 0.3])
    def collision(self):
        for obs in self.obstacle_id:
            contact = p.getContactPoints(bodyA=self.rid, bodyB=obs)
            if contact:
                return True
        return False
    def step(self, action=None, vel=False, discr=False):
        self.prevObs = self.getObs()
        if action is None:
            input = self.getFivePoint()
            angle, distance = self.goalAngle()
            v1, v2 = self.getVel()
            input = np.concatenate([input, [angle, distance, v1, v2]])
            reward = self.rewardF(False)
            return input, reward, False
        if not discr:
            t0, t1 = action
            if not vel:
                # continuous action force control
                p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=t0)
                p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=t1)
            else:
                t0 *= 20
                t1 *= 20
                # continuous action velocity control
                p.setJointMotorControl2(self.rid, 0, p.VELOCITY_CONTROL, targetVelocity=t0, force=100)
                p.setJointMotorControl2(self.rid, 1, p.VELOCITY_CONTROL, targetVelocity=t1, force=100)
        elif not vel:
                # discrete action force control
            if action == 0:
                p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=1)
                p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=1)
                for i in range(2):
                    p.stepSimulation()
            elif action == 3:
                # implement braking
                if self.prevObs[1][0] > 0:
                    p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=-1)
                else:
                    p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=1)
                if self.prevObs[1][1] > 0:
                    p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=-1)
                else:
                    p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=1)
            elif action == 2:
                p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=-1)
                p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=1)
            elif action == 1:
                p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=1)
                p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=-1)
            elif action == 4:
                p.setJointMotorControl2(self.rid, 0, p.TORQUE_CONTROL, force=0)
                p.setJointMotorControl2(self.rid, 1, p.TORQUE_CONTROL, force=0)
        else:
            # discrete action velocity control
            if action == 0:
                p.setJointMotorControl2(self.rid, 0, p.VELOCITY_CONTROL, targetVelocity=60, force=30)
                p.setJointMotorControl2(self.rid, 1, p.VELOCITY_CONTROL, targetVelocity=60, force=30)
            elif action == 1:
                p.setJointMotorControl2(self.rid, 0, p.VELOCITY_CONTROL, targetVelocity=-60, force=30)
                p.setJointMotorControl2(self.rid, 1, p.VELOCITY_CONTROL, targetVelocity=-60, force=30)
            elif action == 2:
                p.setJointMotorControl2(self.rid, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=30)
                p.setJointMotorControl2(self.rid, 1, p.VELOCITY_CONTROL, targetVelocity=10, force=30)
            elif action == 3:
                p.setJointMotorControl2(self.rid, 0, p.VELOCITY_CONTROL, targetVelocity=10, force=30)
                p.setJointMotorControl2(self.rid, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=30)
            for i in range(4):
                p.stepSimulation()
        #if self.visuals:
            #p.resetDebugVisualizerCamera(cameraDistance=4, cameraYaw=ry*180/np.pi-90, cameraPitch=-75, cameraTargetPosition=(x, y, z))
            #time.sleep(0.001)
        p.stepSimulation()
        collision, terminated = self.termination()
        done = terminated or collision
        input = self.getFivePoint()
        angle, distance = self.goalAngle()
        v1, v2 = self.getVel()
        input = np.concatenate([input, [angle, distance, v1, v2]])
        reward = self.rewardF(collision, True, True, False)
        self.steps += 1
        return input, reward, done
    def rewardF(self, collision, distG=True, angle=True, safety=True):
        reward = 0
        obs = self.getObs()
        if distG:
            distR = self.prevObs[3][1] - obs[3][1]
            reward += distR
           # reward -= obs[3][1]
        if angle:
            angR = abs(self.prevObs[3][0])-abs(obs[3][0])
            reward += 10 * angR
           # reward -= abs(obs[3][0])
        if safety:
            if min(obs[0]) < 0.05:
                reward -= 1
        arrival_reward = 1000 if obs[3][1] < self.thresh else 0 
        time_penalty = .2
        reward += arrival_reward
        reward -= time_penalty
        #if collision:
        #    reward -= 200
        return reward
    def termination(self):
        collision = False
        terminated = False
        x, y, z, rr, rp, ry = self.getPos()
        distance = dist((x,y), self.goal)
        # robot leaves ground
        if z > 0.1 or z < -0.5:
            collision = True
        if abs(rr) > 10 or abs(rp) > 10:
            collision = True
        if self.collision():
            collision = True
        if distance < self.thresh:
            terminated = True
        return collision, terminated
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
    def getFivePoint(self):
        x, y, z, rr, rp, ry = self.getPos()
        ry += np.pi/2
        sensor_pos = (x, y, 0.9)
        num_rays = self.lidar
        max_range = 5
        angle = 2* np.pi / num_rays
        data = []
        dest = [x + max_range * np.cos(ry), y + max_range * np.sin (ry), 0.9]
        ray = p.rayTest(sensor_pos, dest)
        data.append(ray[0][2] * max_range if ray else max_range)
        dest = [x + max_range * np.cos(ry - np.pi/4), y + max_range * np.sin (ry - np.pi/4), 0.9]
        ray = p.rayTest(sensor_pos, dest)
        data.append(ray[0][2] * max_range if ray else max_range)
        dest = [x + max_range * np.cos(ry + np.pi/4), y + max_range * np.sin (ry - np.pi/4), 0.9]
        ray = p.rayTest(sensor_pos, dest)
        data.append(ray[0][2] * max_range if ray else max_range)
        dest = [x + max_range * np.cos(ry - np.pi/2), y + max_range * np.sin (ry - np.pi/2), 0.9]
        ray = p.rayTest(sensor_pos, dest)
        data.append(ray[0][2] * max_range if ray else max_range)
        dest = [x + max_range * np.cos(ry + np.pi/2), y + max_range * np.sin (ry + np.pi/2), 0.9]
        ray = p.rayTest(sensor_pos, dest)
        data.append(ray[0][2] * max_range if ray else max_range)
        #print(data)
        return data
    def goalAngle(self):
        x, y, z, rr, rp, ry = self.getPos()
        gx, gy = self.goal
        dx = gx - x
        dy = gy - y
        a_world = math.atan2(dy, dx)
        a_relative = a_world - ry
    
        # Normalize the angle to be within the range [-pi, pi]
        if a_relative > math.pi:
            a_relative -= 2 * math.pi
        elif a_relative < -math.pi:
            a_relative += 2 * math.pi    
        distance = dist((x,y), self.goal)
        return (a_relative, distance)
    def goalWorldAngle(self, x, y):
        gx, gy = self.goal
        dx = gx - x
        dy = gy - y
        a_world = math.atan2(dy, dx)
        return a_world
    def reset(self):
        if p.isConnected():
            p.disconnect()
        if self.visuals:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.goalmarker = -1
        id = p.loadURDF('assets/ground.urdf', [0, 0, -0.1])
        p.setGravity(0, 0, -9.81) 
        self.rid = self.loadAgent(-5, -5)
        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=90, cameraPitch=-75, cameraTargetPosition=(2, 2, 0))
        self.steps = 0
    def lidarParse(self, data):
        n = len(data)
        angle = 2* np.pi / n
        x, y, z, rr, rp, ry = self.getPos()
        x = [d * np.cos(i * angle) for i, d in enumerate(data)]
        y = [d * np.sin(i * angle) for i, d in enumerate(data)]
        if n == 5:
            ry -= np.pi/2
            x = [data[0] * np.cos(ry), data[2] * np.cos(ry - np.pi/4), data[1] * np.cos(ry + np.pi/4), data[4] * np.cos(ry - np.pi/2), data[3] * np.cos(ry + np.pi/2)]
            y = [data[0] * np.sin(ry), data[2] * np.sin(ry - np.pi/4), data[1] * np.sin(ry + np.pi/4), data[4] * np.sin(ry - np.pi/2), data[3] * np.sin(ry + np.pi/2)]
        return x, y
    def lidarGraph(self, data):
        x, y = self.lidarParse(data)
        for i in range(len(x)):   
            #plt.plot(x[i], y[i], 'o')
            plt.plot([0, x[i]], [0, y[i]])
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('lidar.png')
        plt.show()