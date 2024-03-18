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

def showPRM(args):
    map = prm.PRM()
    map.env.load('./env_0.txt')
    time1 = time.time()
    map.plan(False, True)
    time2 = time.time()
    print(time2 - time1)
    map.visualize()
    print(map.graph.size)
    #print(map.graph.e)
    a = prm.Node(0.9648, 1.1297)
    b = prm.Node(1.3363, 0.8655)
    #map.addNode(a)
    #map.addNode(b)
    #map.addEdge(0, 1)
    #map.visualize()
    #print(map.steerTo(a, b))
    #for i in range(map.graph.size):
    #   print(map.getNode(i).x, map.getNode(i).y)


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