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

def showPath(path):
    if path is None:
        print("no path found")
        return
    for n in range(1, len(path)):
        plt.plot([path[n].x, path[n-1].x],[path[n].y, path[n-1].y], '-or')

def showPRM(args):
    map = prm.PRM(tree=True)
    map.env.load('./env_0.txt')
    time1 = time.time()
    map.plan(500, True, True, 2)
    time2 = time.time()
    print(time2 - time1)
    map.visualize()
    print(map.graph.size)
    path, cost = map.getPath((0.5, 9.5), (9.5, 0.5))
    showPath(path)
    print(cost)


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