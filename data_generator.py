import prm
import os
import numpy as np

def savePath(path, dir, filepath):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filepath, "w") as f:
        for node in path:
            line = str(node.x) + ' ' + str(node.y) + '\n'
            f.write(line)

root = './data/env'
num_envs = 1
start_env = 1
num_paths = 20
for i in range(start_env, start_env + num_envs):
    env_path = './data/envs/env_' + str(i) + '.txt'
    map = prm.PRM(tree=False, geom='circle')
    map.env.load(env_path)
    # get prm
    size = 500
    map.plan(size, False, True, 2)
    while not map.cleanRegions():
        size *= 2
        map.plan(size, False, True, 2)
    dir = str(root) + str(i)
    prm_path = '/prm.txt'
    map.save(dir, prm_path)
    count = 0
    while count < num_paths:
        x = np.random.uniform() * map.env.length
        y = np.random.uniform() * map.env.width
        while map.collision((x, y)):
            x = np.random.uniform() * map.env.length
            y = np.random.uniform() * map.env.width
        start = (x, y)
        x = np.random.uniform() * map.env.length
        y = np.random.uniform() * map.env.width
        while map.collision((x, y)):
            x = np.random.uniform() * map.env.length
            y = np.random.uniform() * map.env.width
        goal = (x, y)
        path, cost = map.getPath(start, goal)
        if path is not None:
            filepath = str(root) + str(i) + '/path' + str(count) + '.txt'
            savePath(path, dir, filepath)
            count += 1

