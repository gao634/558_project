import prm
import os

root = './data/env'
num_envs = 100
for i in range(num_envs):
    env_path = './data/envs/env_' + str(i) + '.txt'
    map = prm.PRM()
    map.env.load(env_path)
    # get prm
    map.plan(500, False, True, 2)
    prm_path = str(root) + str(i) + 'prm.txt'
    map.save()
