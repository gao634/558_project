import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import copy

# taken from assignment 1 part 2
def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

def distNode(n1, n2):
    return dist((n1.x, n1.y), (n2.x, n2.y))
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
    def visualize(self):
        # for now, our maze runs on grid coordinates and each obs is a 1x1 square
        for x in range(self.length):
            for y in range(self.width):
                if self.obs[x][y]:
                    rect = mpatches.Rectangle((x, y), 1, 1, color='grey')
                    plt.gca().add_patch(rect)

# this class creates the road map. one rm is created per env to generate paths
class PRM():
    def __init__(self, env=ENV(), step_size=0.05, n_iters = 100):
        self.env = env
        self.step_size = step_size
        self.graph = Graph()
        self.n_iters = n_iters
    def plan(self, animate=False):
        for i in range(self.n_iters):
            print(i)
            node = self.generateSample()
            if self.collision(node.coord()):
                continue
            near = self.getNearest(node)
            if near is None:
                self.addNode(node)
                continue
            if self.steerTo(self.getNode(near), node):
                index = self.addNode(node)
                self.addEdge(near, index)
                #print(near)
                if animate:
                    self.visualize(node)
    # returns true if no collision
    """
    def steerTo(self, start, goal):
        dir = diff(start.coord(), goal.coord())
        # normalize direction vector
        distance = distNode(start, goal)
        dir = [x * self.step_size / distance for x in dir]
        n = int(math.floor(distance / self.step_size)) + 1
        for i in range(n):
            temp = (start.x + dir[0] * i, start.y + dir[1] * i)
            if self.collision(temp):
                return False
        return True
    """
    def steerTo(self, dest, source):
        newNode = copy.deepcopy(source)

        DISCRETIZATION_STEP=self.step_size

        dists = [dest.x - source.x, dest.y - source.y]

        distTotal = magnitude(dists)


        if distTotal>0:
            incrementTotal = distTotal/DISCRETIZATION_STEP
            for j in range(0, 2):
                dists[j] =dists[j]/incrementTotal
            numSegments = int(math.floor(incrementTotal))+1

            stateCurr = np.zeros(2,dtype=np.float32)
            for j in range(0,2):
                stateCurr[j] = newNode.coord()[j]

            for i in range(0,numSegments):

                if not self.collision(stateCurr):
                    return (False, None)

                for j in range(0,2):
                    stateCurr[j] += dists[j]

            if not self.collision(dest.coord()):
                return False

            return True
        else:
            return False
    # returns false if no collision
    def collision(self, pos):
        x, y = pos
        if x > self.env.length or y > self.env.width:
            return True
        if x < 0 or y < 0:
            return True
        for i in range(self.env.length):
            for j in range(self.env.width):
                if self.env.obs[i][j]:
                    if abs(i - x + 0.5) < 0.5 and abs(j - y + 0.5) < 0.5:
                        return True
        return False      
    def generateSample(self):
        x = np.random.uniform(0, self.env.length)
        y = np.random.uniform(0, self.env.width)
        return Node(x, y)
    def addNode(self, node):
        index = self.graph.size
        self.graph.size += 1
        self.graph.v.append(node)
        return index
    def addEdge(self, a, b):
        self.graph.e.append((a, b))
    def getNode(self, index):
        return self.graph.v[index]
    def getNears(self, node):
        if not self.graph.size:
            return None
        # dist function tbd
        thresh = 5
        nears = []
        for i in range(self.graph.size):
            if distNode(node, self.getNode(i)) < thresh:
                nears.append(i)
        return nears
    def getNearest(self, node):
        if not self.graph.size:
            return None
        thresh = distNode(node, self.getNode(0))
        near = 0
        for i in range(self.graph.size):
            if distNode(node, self.getNode(i)) < thresh:
                near = i
        return near
    # saves the graph of the road map so it can be used later to generate paths
    def save():
        pass
    # loads graph from data file for visuals
    def load():
        pass
    def visualize(self, node=None):
        self.env.visualize()
        for edge in self.graph.e:
            plt.plot([self.getNode(edge[0]).x, self.getNode(edge[1]).x],
                    [self.getNode(edge[0]).y, self.getNode(edge[1]).y], '-g')
        if node is not None:
            plt.plot(node.x, node.y, 'k')
        plt.axis("equal")
        plt.axis([0, 10, 0, 10])
        plt.grid(True)
        plt.pause(0.01)

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
    def coord(self):
        return (self.x, self.y)