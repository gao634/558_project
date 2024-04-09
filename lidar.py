import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

def getInput(id, size):
    pos = pos, orn = p.getBasePositionAndOrientation(id)
    x, y, z = pos
    rr, rp, ry = p.getEulerFromQuaternion(orn)
    sensor_pos = (x, y, 0.9)
    num_rays = size
    max_range = 5
    angle = 2* np.pi / num_rays
    data = []
    for i in range(num_rays):
        dest = [x + max_range * np.cos(i * angle), y + max_range * np.sin (i * angle), 0.9]
        ray = p.rayTest(sensor_pos, dest)
        data.append(ray[0][2] * max_range if ray else max_range)
    return data

def lidarGraph(data):
    n = len(data)
    angle = 2* np.pi / n
    x = [d * np.cos(i * angle) for i, d in enumerate(data)]
    y = [d * np.sin(i * angle) for i, d in enumerate(data)]
    for i in range(n):   
        plt.plot(x[i], y[i], 'o')
    plt.grid(True)
    plt.show()
