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

# force quits program when x button is pressed on animation
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
    def __init__(self, env=ENV(), step_size=0.01, tree=False):
        self.env = env
        self.step_size = step_size
        self.graph = Graph()
        # determines if the prm is a graph or tree structure
        # get nearest will return a single node if tree, multiple if graph
        # if tree, graph.e and graph.regions will be empty, node.cost will be distance from root
        # if graph, node.parent and node.children will be empty
        self.tree = tree
    def plan(self, n_iters=1000, animate=False, space_saving=True, split=2):
        for i in range(n_iters):
            node = self.generateSample()
            if self.collision(node.coord()):
                continue
            nears = self.getNearest(node, split, space_saving)
            # if graph is empty
            if nears is None:
                self.addNode(node)
                if animate:
                    self.visualize(node, i)
                continue
            # if graph, add the node even if can't steer for now
            if not self.tree:
                if space_saving and len(nears) > 0:
                    if nears[0] == -1:
                        continue
                index = self.addNode(node)
                if animate:
                    self.visualize(node, i)
            for near in nears:
                if self.tree:
                    index = self.addNode(node)
                self.addEdge(near, index)
                if animate:
                    self.visualize(node, i)
        print("planning complete")
    # connect regions, called outside of loop to reduce computation costs
    def cleanRegions(self):
        pass
    def getPath(self, start_coord, goal_coord):
        # if coords are in collision
        if self.collision(start_coord) or self.collision(goal_coord):
            return None, 0
        start = Node(start_coord[0], start_coord[1])
        goal = Node(goal_coord[0], goal_coord[1])
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
                    start_nearest_dist = cost
            if self.steerTo(v, goal):
                cost = distNode(v, goal)
                if cost < goal_nearest_dist:
                    goal_nearest = v
                    goal_nearest_dist = cost
        # if road map sucks
        if not start_nearest or not goal_nearest:
            return None, 0
        path = [start]
        cost = 0
        # path finding for tree structure
        if self.tree:
            # all nodes lead to root, then from root go up until find divergence
            start_path = []
            goal_path = []
            temp = start_nearest
            while temp is not self.getNode(0):
                start_path.append(temp)
                temp = self.getNode(temp.parent)
            temp = goal_nearest
            while temp is not self.getNode(0):
                goal_path.append(temp)
                temp = self.getNode(temp.parent)
            start_path.append(self.getNode(0))
            goal_path.append(self.getNode(0))
            #check divergence
            for i in range(self.graph.size):
                if start_path[-1-i] is not goal_path[-1-i]:
                    diverge = i
                    start_cost = start_nearest.cost - start_path[-1-i].cost
                    goal_cost = goal_nearest.cost - goal_path[-1-i].cost
                    break
            for i in range(len(start_path)-diverge):
                path.append(start_path[i])
            for i in range(len(goal_path)-diverge):
                path.append(goal_path[len(goal_path)-diverge-i])
            path.append(goal)
            cost = goal_cost + start_cost + start_nearest_dist + goal_nearest_dist
        # path finding for graph (dijkstras)
        else:
            pass
        return path, cost
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
        if not self.tree:
            self.graph.regions.append([index])
        return index
    def addEdge(self, a, b):
        cost = distNode(self.getNode(a), self.getNode(b))
        if self.tree:
            self.getNode(a).children.add(b)
            self.getNode(b).parent = a
            self.getNode(b).cost = cost + self.getNode(a).cost
        else:
            self.graph.e.append((a, b, cost))
            for region in self.graph.regions:
                # combine regions
                if a in region and b not in region:
                    # find b region
                    for b_region in self.graph.regions:
                        if b in b_region:
                            combined = region + b_region
                            self.graph.regions.remove(region)
                            self.graph.regions.remove(b_region)
                            self.graph.regions.append(combined)
    def getNode(self, index):
        return self.graph.v[index]
    def getNearest(self, node, split, space_saving):
        if not self.graph.size:
            return None
        dlist = []
        for n in self.graph.v:
            dlist.append(distNode(node, n))
        min_dist = min(dlist)
        near = dlist.index(min_dist)
        if self.tree:
            if space_saving and min_dist < 0.5:
                return []
            if self.steerTo(self.getNode(near), node):
                return [near]
            return []
        # for graph structure in order to minimize edges, only closest few will be connected
        # however, we want isolated regions to connect, so we check every other region as well
        nears = []
        # connect to closest node to each region
        for region in self.graph.regions:
            min_dist = float('inf')
            nearest = -1
            for n in region:
                distance = distNode(node, self.getNode(n))
                if distance < min_dist:
                    if (space_saving and distance > 0.5) or not space_saving:
                        min_dist = distance
                        nearest = n
            if nearest != -1 and self.steerTo(node, self.getNode(nearest)):
                nears.append(nearest)
        # check [split] closest nodes
        sorted_dlist = []
        for i in range(self.graph.size):
            sorted_dlist.append((dlist[i], i))
        sorted_dlist.sort()
        for i in range(min(split, self.graph.size)):
            dist, ind = sorted_dlist[i]
            if ind not in nears:
                if self.steerTo(self.getNode(ind), node):
                    if (space_saving and dist > 0.5) or not space_saving:
                        nears.append(ind)
        # if closest node is within 0.5 on space saver, add flag so node is not added
        if space_saving and sorted_dlist[0][0] < 0.5:
            nears.insert(0, -1)
        return nears
    # saves the graph of the road map so it can be used later to generate paths
    def save(self, file_path):
        pass
    # loads graph from data file for visuals
    def load(self, file_path):
        pass
    def visualize(self, node=None, iter=-1):
        plt.clf()
        self.env.visualize()
        if iter >= 0:
            plt.text(-1, -1, 'n=' + str(iter), fontsize=10, color='blue')
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
            plt.plot([self.getNode(i).x], [self.getNode(i).y], '-og')
            plt.text(self.getNode(i).x + 0.03, self.getNode(i).y + 0.03, str(i), fontsize=10, color='blue')
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
        self.regions = []

# node data structure
class Node():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.children = set()
    def coord(self):
        return (self.x, self.y)