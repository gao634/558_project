import pybullet as p
physics_client = p.connect(p.GUI)
goal_marker = p.loadURDF('assets/marker.urdf', [0, 0, 0])
while True:
    p.stepSimulation()