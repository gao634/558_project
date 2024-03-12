import math

# taken from assignment 1 part 2
def diff(v1, v2):
    return [x1 - x2 for x1, x2 in zip(v1, v2)]

def magnitude(v):
    return math.sqrt(sum([x*x for x in v]))

def dist(p1, p2):
    return magnitude(diff(p1, p2))

# this class creates the road map
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

# graph data structure
class Graph():
    def __init__():
        pass
    def load():
        pass
    def visualize():
        pass
