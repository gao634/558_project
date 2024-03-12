import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
import prm

def showPRM(args):
    map = prm.PRM()
    map.env.load('./env_0.txt')
    map.plan(True)
    map.visualize()
    print(map.graph.size)
    print(len(map.graph.e))
    print(map.graph.e)


def main(args):
    showPRM(args)
    plt.show()

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