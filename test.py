import pybullet as p
physics_client = p.connect(p.GUI)
p.loadURDF("./turtlebot-main/final_challenge/assets/cube.urdf", [0, -1, 0.5])
p.loadURDF("./turtlebot-main/data/turtlebot.urdf")
while True:
    # Step the simulation
    p.stepSimulation()