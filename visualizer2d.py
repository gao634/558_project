import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import prm
import time

# kinda sucks, will make them by hand for now
def generateENV(l, w, density):
    env = np.zeros(l, w)
    for i in range(l):
        for j in range(w):
            rnd = np.random.uniform(0, 1)
            if rnd < density:
                env[i][j] = 1

def getPathData(file):
    data = np.loadtxt(file, np.float32)
    return data
def loadPath(file):
    data = getPathData(file)
    path = []
    for point in data:
        path.append(prm.Node(point[0], point[1]))
    return path
def showPath(path):
    if path is None:
        print("no path found")
        return
    #plt.text(path[0].x - 0.5, path[0].y + 0.03, '0', fontsize=10, color='green')
    for n in range(1, len(path)):
        plt.plot([path[n].x, path[n-1].x],[path[n].y, path[n-1].y], '-or')
        #plt.text(path[n].x - 0.5, path[n].y + 0.03, str(n), fontsize=10, color='green')

def showPRM(args):
    map = prm.PRM(tree=False, geom='circle')
    map.env.load('data/envs/env_0.txt')
    map.load('data/env0/prm.txt')
    time1 = time.time()
    #map.plan(500, False, True, 2)
    #map.save('test_prm.txt')
    time2 = time.time()
    #print(time2 - time1)
    map.visualize()
    print(map.graph.size)
    #path, cost = map.getPath((0.5, 9.5), (9.5, 0.5))
    path = loadPath('data/env0/path1.txt')
    showPath(path)
    #print(cost)


def main(args):
    
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('close_event', prm.cleanup)
    showPRM(args)
    plt.show()

# args not used for now
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='./data/')
parser.add_argument('--env-id', type=int, default=0)
parser.add_argument('--path-id', type=int, default=2000)
parser.add_argument('--cmp', default=False, action='store_true')
parser.add_argument('--single-path', default=False, action='store_true')
parser.add_argument('--point-cloud', default=False, action='store_true')
parser.add_argument('--path-file', nargs='*', type=str, default=[], help='path file')
args = parser.parse_args()

print(args)
main(args)