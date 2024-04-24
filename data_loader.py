import numpy as np

def dataloader(start_env, num_envs, start_path, num_paths):
    data = []
    root = 'data/env'
    for i in range(start_env, start_env + num_envs):
        for j in range(start_path, start_path + num_paths):
            file = str(root) + str(i) + '/path' + str(j) + '.txt'
            path = np.loadtxt(file, np.float32)
            for n in range(1, len(path)):
                segment = [(path[n-1][0], path[n-1][1]),(path[n][0], path[n][1])]
                data.append(segment)
    return data

