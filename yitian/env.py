import time
import pybullet as p
import pybullet_data
import numpy as np
import math

class MazeEnv:
    def __init__(self, render=True):
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        self.goal_position = np.array([2, -1, 0])  # Example goal position in the maze
        self.goalmarker = p.loadURDF('assets/marker.urdf', [2, -1, 0.3])
        self.reset_before_step = False
        # self.reset()

    def load_environment(self):
        """ Load the maze and static elements here """
        planeId = p.loadURDF("yitian/plane.urdf")
        p.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.01)
        mazeId = p.loadURDF("yitian/maze.urdf", basePosition=[0, -1, -0.7])  # Assuming a URDF for the maze

    def load_robot(self):
        """ Initialize and return the robot in the environment """
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        ret = p.loadURDF("yitian/two_wheeled_robot.urdf", [0, 0, 1.1], start_orientation)
        p.changeDynamics(ret, linkIndex=-1, lateralFriction=0.2)
        return ret

    def reset(self):
        """ Reset the environment and the robot for a new episode """
        self.reset_before_step = True
        
        p.resetSimulation(self.client)
        p.setTimeStep(1 / 20)
        p.setGravity(0, 0, -10)
        self.goalmarker = p.loadURDF('assets/marker.urdf', [2, -1, 0.3])
        self.load_environment()
        self.robot_id = self.load_robot()
        obs = self.get_observation()
        self.past_observation = obs
        self.steps = 0
        
        
        return obs

    def get_observation(self):
        """ Get the state of the robot including raycast data and goal distance """
        
        assert self.reset_before_step
        
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        position = np.array(position)
        
        euler = p.getEulerFromQuaternion(orientation)
        robot_yaw = euler[2]

        # Raycasting to detect walls
        self.num_rays = 5
        ray_len = 4
        angles = np.linspace(-np.pi / 4, np.pi / 4, self.num_rays)
        ray_from = position
        ray_from[0] += 0.9
        ray_to = []

        # Calculate ray destinations considering the robot's orientation
        for angle in angles:
            world_angle = angle + robot_yaw  # Adjust angle based on the robot's current yaw
            dx = ray_len * math.cos(world_angle)
            dy = ray_len * math.sin(world_angle)
            ray_to.append([ray_from[0] + dx, ray_from[1] + dy, position[2]])

        # Perform ray tests
        hit_distances = []
        for r in ray_to:
            # p.addUserDebugLine(ray_from, r, [1, 0, 0])
            results = p.rayTest(ray_from, r, self.client)
            hit_fraction = results[0][2]  # Fraction of the ray's total length that was hit
            if hit_fraction == 1.0:  # No hit
                hit_distances.append(ray_len)  # Use maximum ray length if no hit
            else:
                hit_distance = hit_fraction * ray_len
                hit_distances.append(hit_distance)
                
        # Distance and angle to the goal
        goal_direction = self.goal_position - position
        goal_distance = np.linalg.norm(goal_direction[0:2])
        goal_angle = math.atan2(goal_direction[1], goal_direction[0])
        
        heading_deviation = goal_angle - robot_yaw
        # Normalize the angle to be within -pi to pi
        heading_deviation = (heading_deviation + np.pi) % (2 * np.pi) - np.pi
        
        # print('dddddd', math.degrees(heading_deviation), math.degrees(goal_angle), math.degrees(robot_yaw))

        # Debugging output
        # print("Goal distance: ", goal_distance)
        # print("Goal angle (degrees): ", math.degrees(goal_angle))
        # print("Robot orientation (yaw, degrees): ", math.degrees(robot_yaw))
        # print("Heading deviation (degrees): ", math.degrees(heading_deviation))

        obs = np.concatenate([hit_distances, [goal_distance, heading_deviation]])
        return obs

    def step(self, action):
        """ Apply an action and return the new state, reward, and done status """
        
        assert self.reset_before_step
        
        desired_velocity, turn_vel = 0, 0
        if action == 0:  # Move forward
            forward_vel = 8
            turn_vel = 0
        elif action == 1:  # Turn left
            forward_vel = 0
            turn_vel = 0.7
        elif action == 2:  # Turn right
            forward_vel = 0
            turn_vel = -0.7
        elif action == 3:  # Back
            forward_vel = -8
            turn_vel = 0
        
        # Simple P controller
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        _, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_yaw = p.getEulerFromQuaternion(orientation)[2]  # Yaw is the rotation around the Z-axis

        # Calculate the forward direction based on the yaw
        forward_dir = np.array([math.cos(robot_yaw), math.sin(robot_yaw)])
        
        # Compute control inputs (force along the direction the robot is facing)
        current_velocity = np.array(linear_velocity[:2])  # Ignore Z component
        desired_velocity = forward_vel * forward_dir
        force = 3 * (desired_velocity - current_velocity)  # Proportional control with gain
        torque = 3 * (turn_vel - angular_velocity[2])

        # Apply the calculated force and torque to the robot
        # Convert force back to tuple to use in PyBullet
        p.applyExternalForce(self.robot_id, -1, force.tolist() + [0], [0, 0, 0], p.LINK_FRAME)
        p.applyExternalTorque(self.robot_id, -1, [0, 0, torque], p.LINK_FRAME)

        # Step simulation
        for _ in range(10):
            p.stepSimulation()
        # 
        # time.sleep(1/30)
        
        # Get new observation
        next_state = self.get_observation()
        
        heading_index = self.num_rays + 1
        dist_index = self.num_rays
        goal_dist = next_state[dist_index]
        
        heading_reward = abs(self.past_observation[heading_index]) - abs(next_state[heading_index])
        dist_reward = self.past_observation[dist_index] - next_state[dist_index]
        
        # print(dist_reward, heading_reward)
                
        self.past_observation = next_state
        
        arrival_reward = 20 if goal_dist < 0.4 else 0 
        time_penalty = 0.05 # Existence is pain
                
        # Check if goal is reached
        done = goal_dist < 0.4 or goal_dist > 6
        
        self.steps += 1

        return next_state, 0.5 * heading_reward + dist_reward + arrival_reward - time_penalty, done

    def close(self):
        """ Clean up the environment """
        p.disconnect(self.client)
