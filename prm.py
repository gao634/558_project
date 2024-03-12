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
    def __init__(self, env, step_size=0.01, n_iters = 100):
        self.env = env
        self.step_size = step_size
        self.nodes = Graph()
        self.n_iters = n_iters
    def plan(self, animate=False):
        for i in range(self.n_iters):
            node = self.generateSample()
            if self.graph.size == 0:
                self.addNode(node)
            nears = self.getNearest(node)
            for near in nears:
                added = False
                if not self.steerTo(self.getNode(near), node):
                    if not added:
                        added = True
                        index = self.addNode(node)
                    self.addEdge(near, index)
                    if animate:
                        plt.plot([self.getNode(near).x, self.getNode(index).x],
                            [self.getNode(near).y, self.getNode(index).y], '-g')
    def steerTo(self, start, goal):
        diff = diff(start, goal)
        diff *= self.step_size
        for i in range(1/self.step_size):
            temp = (start[0] + diff[0], start[1] + diff[1])
            if self.collision(self, temp):
                return False
        return True
    # returns false if no collision
    def collision(self, pos):
        x, y = pos
        if x > self.env.length or y > self.env.width:
            return True
        for obstacle in self.env.obs:
            collision = True
            if abs(obstacle[0] - x + 0.5) > 0.5:
                collision = False
            elif abs(obstacle[1] - y + 0.5) > 0.5:
                collision = False
        return collision      
    def generateSample(self):
        x = np.random.uniform(0, self.env.length)
        y = np.random.uniform(0, self.env.width)
        return (x, y)
    def addNode(self, node):
        self.graph.size += 1
        index = len(self.graph.v)
        self.graph.v.append(node)
        return index
    def addEdge(self, a, b):
        self.graph.e.append((self.getNode(a), self.getNode(b)))
    def getNode(self, index):
        return self.graph.v[index]
    def getNearest(self, node):
        # dist function tbd
        range = 5
        nears = []
        for i in range(self.graph.size):
            if dist(node, self.getNode(i)) < range:
                nears.append(i)
        return nears
    # saves the graph of the road map so it can be used later to generate paths
    def save():
        pass
    # loads graph from data file for visuals
    def load():
        pass
    def visualize(self):
        for edge in self.graph.e:
            plt.plot([self.getNode(edge[0]).x, self.getNode(edge[1]).x],
                    [self.getNode(edge[0]).y, self.getNode(edge[1]).y], '-g')

# graph data structure
class Graph():
    def __init__(self):
        self.size = 0
        self.v = []
        self.e = []

# node data structure
class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y

env = ENV()
env.load('./env_0.txt')