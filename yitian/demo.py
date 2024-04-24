import pybullet as p
import time
import pybullet_data

def main():
    # Connect to Physics Server
    physicsClient = p.connect(p.GUI)  # p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Optionally

    # Set Gravity
    p.setGravity(0, 0, -9.81)

    # Load ground plane
    ground_id = p.loadURDF("plane.urdf")

    # Load a simple box
    box_start_position = [0, 0, 1]
    box_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    # box_id = p.loadURDF("r2d2.urdf", box_start_position, box_start_orientation)
    box_id = p.loadURDF("two_wheeled_robot.urdf", box_start_position, box_start_orientation)
    # Simulation loop
    for i in range(1000):
        p.stepSimulation()
        time.sleep(1./240)  # Time step the simulation at 240 Hz
        if i % 240 == 0:  # Apply force every second
            # Apply a force to the box
            p.applyExternalForce(objectUniqueId=box_id, linkIndex=-1, forceObj=[100, 0, 0], posObj=[0, 0, 0], flags=p.WORLD_FRAME)

        # Optional: print the box's position and orientation
        pos, orn = p.getBasePositionAndOrientation(box_id)
        print("Box position:", pos, "Box orientation:", p.getEulerFromQuaternion(orn))

    p.disconnect()

if __name__ == "__main__":
    main()
