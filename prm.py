import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import copy
import sys

# taken from assignment 1 part 2
def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

def distNode(n1, n2):
    return dist((n1.x, n1.y), (n2.x, n2.y))

def cleanup(event):
    plt.close()
    print("Window Closed")
    sys.exit()

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
    def __init__(self, env=ENV(), step_size=0.01, n_iters = 1000, tree=False):
        self.env = env
        self.step_size = step_size
        self.graph = Graph()
        self.n_iters = n_iters
        # determines if the prm is a graph or tree structure
        # get nearest will return a single node if tree, multiple if graph
        # if tree, graph.e will be empty, node.cost will be distance from root
        # if graph, node.parent and node.children will be empty
        self.tree = tree
    def plan(self, animate=False, space_saving=True):
        for i in range(self.n_iters):
            node = self.generateSample()
            if self.collision(node.coord()):
                continue
            nears = self.getNearest(node)
            # if graph is empty
            if nears is None:
                self.addNode(node)
                continue
            for near in nears:
                if self.steerTo(self.getNode(near), node):
                    if space_saving:
                        if distNode(node, self.getNode(near)) < 0.5:
                            continue
                    index = self.addNode(node)
                    self.addEdge(near, index)
                    if animate:
                        self.visualize(node)
        print("planning complete")
    def getPath(self, start_coord, goal_coord):
        # if coords are in collision
        if self.collision(start_coord) or self.collision(goal_coord):
            return None
        start = Node(start_coord)
        goal = Node(goal_coord)
        # steer to every node and connect to the closest ones
        start_nearest = None
        start_nearest_dist = float('inf')
        goal_nearest = None
        goal_nearest_dist = float('inf')
        for v in self.graph.v:
            if self.steerTo(v, start):
                cost = distNode(v, start)
                if cost < start_nearest_dist:
                    start_nearest = v
                    start_nerest_dist = cost
            if self.steerTo(v, goal):
                cost = distNode(v, goal)
                if cost < goal_nearest_dist:
                    goal_nearest = v
                    goal_nerest_dist = cost
        # if road map sucks
        if not start_nearest or not goal_nearest:
            return None
        # path finding for tree structure
        if self.tree:
            pass
        # path finding for graph (dijkstras)
        else:
            pass


    # returns true if no collision
    def steerTo(self, start, goal):
        dir = diff(goal.coord(), start.coord())
        # normalize direction vector
        distance = distNode(start, goal)
        dir = [x * self.step_size / distance for x in dir]
        n = int(math.floor(distance / self.step_size)) + 1
        for i in range(n):
            temp = (start.x + dir[0] * i, start.y + dir[1] * i)
            if self.collision(temp):
                return False
        return True
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
        cost = distNode(self.getNode(a), self.getNode(b))
        if self.tree:
            self.getNode(a).children.add(b)
            self.getNode(b).parent = a
            self.getNode(b).cost = cost + self.getNode(a).cost
        else:
            self.graph.e.append((a, b, cost))
    def getNode(self, index):
        return self.graph.v[index]
    def getNearest(self, node):
        if not self.graph.size:
            return None
        if self.tree:
            pass
        else:
            thresh = distNode(node, self.getNode(0))
            near = 0
            for i in range(self.graph.size):
                distance = distNode(node, self.getNode(i))
                if distance < thresh:
                    near = i
                    thresh = distance
            return [near]
    # saves the graph of the road map so it can be used later to generate paths
    def save():
        pass
    # loads graph from data file for visuals
    def load():
        pass
    def visualize(self, node=None):
        plt.clf()
        self.env.visualize()
        if self.tree:    
            for node in self.graph.v:
                if node.parent is not None:
                    plt.plot([node.x, self.getNode(node.parent).x], [
                            node.y, self.getNode(node.parent).y], "-g")
        else:
            for edge in self.graph.e:
                plt.plot([self.getNode(edge[0]).x, self.getNode(edge[1]).x],
                        [self.getNode(edge[0]).y, self.getNode(edge[1]).y], '-g')
        for i in range(self.graph.size):
            pass
            #plt.plot([self.getNode(i).x], [self.getNode(i).y],)
            #plt.text(self.getNode(i).x + 0.03, self.getNode(i).y + 0.03, str(i), fontsize=10, color='blue')
        if node is not None:
            plt.plot(node.x, node.y, '^r')
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.set_xlim([0, self.env.length])
        ax.set_ylim([0, self.env.width])
        ax.set_aspect('equal', adjustable='box')
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
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = parent
        self.children = set()
    def coord(self):
        return (self.x, self.y)