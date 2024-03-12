import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# taken from assignment 1 part 2
def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

# this class loads an environment with obstacles to run PRM on
class ENV():
    def __init__(self, length=20, width=20, obs=[]):
        self.length = length
        self.width = width
        self.obs=obs   
    def load(self, path):
        if path.endswith('.txt'):
            self.obs = np.loadtxt(path, np.int32)
        else:
            self.obs = np.fromfile(path, np.int32)
        self.length, self.width = self.obs.shape
    # takes a plt as input so it can be called in visualizer.py
    def visualize(self, plt):
        # for now, our maze runs on grid coordinates and each obs is a 1x1 square
        for x in self.obs:
            for y in self.obs[0]:
                if self.obs[x][y]:
                    rect = mpatches.Rectangle((x, y), 1, 1, color='grey')
                    plt.gca().add_patch(rect)
        return plt

# this class creates the road map. one rm is created per env to generate paths
class PRM():
    def __init__():
        pass
    def steerTo():
        pass
    def collision():
        pass
    def generateSample():
        pass
    def addNode():
        pass
    def getNearest():
        pass
    # saves the graph of the road map so it can be used later to generate paths
    def save():
        pass
    # loads graph from data file for visuals
    def load():
        pass
    def visualize():
        pass

# graph data structure
class Graph():
    def __init__():
        pass

env = ENV()
env.load('./env_0.txt')