import numpy as np
import rclpy
from rclpy.node import Node
from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from lab05_pkg.utils import normalize_angle
from std_msgs.msg import Float32

#TODO Real robot: change topic names (landmarks to /camera/landmarks).....DONE
#TODO Provide intermediate feedback (distance to goal every 50 steps) in a dedicated topic
#TODO Revise max velocities, weights, and safety distances for real robot

# Topic names:
TOPIC_LANDMARKS = '/camera/landmarks'
TOPIC_ODOM = '/odom'
TOPIC_LASER = '/scan'
TOPIC_DGP = '/dynamic_goal_pose'  # For simulation only


class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('Obstacle_Avoidance_Node')

        # Timer for controller callback
        self.timer_period = 1/15  # seconds
        self.timer = self.create_timer(self.timer_period, self.go_to_pose)
        
        # Subscriptions
        self.create_subscription(LandmarkArray, TOPIC_LANDMARKS, self.landmark_callback, 10) # TODO Robot reale: /camera/landmarks
        self.create_subscription(Odometry, TOPIC_ODOM, self.odom_callback, 10)
        self.create_subscription(LaserScan, TOPIC_LASER, self.laser_callback, 10)
        self.create_subscription(Float32, '/goal_feedback', self.goal_feedback_callback, 10)

        # Simulation subscription
        self.create_subscription(Odometry, TOPIC_DGP, self.dynamic_goal_callback, 10)

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_feedback_pub = self.create_publisher(Float32, '/goal_feedback', 10)

        # General variables
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.goal_pose = np.array([0.0, 0.0])
        self.lasers = None  # To store laser scan data
        self.angle_min=0
        self.angle_max=0
        self.angle_increment=0


        #All of the following constants can be tuned for better performance and for real robot use
        # Constants
        self.GOAL_TOLERANCE = 0.2  # meters
        self.LASERS_OBS_NUM = 30  # Number of laser readings to consider for obstacle avoidance
        self.VMAX = 0.5  # Maximum linear velocity (m/s)
        self.WMAX = 1.5  # Maximum angular velocity (rad/s)
        self.V_STEPS = 10  # Number of linear velocity samples
        self.W_STEPS = 15  # Number of angular velocity samples
        self.MAX_LASER_RANGE = 3.5  # Maximum laser range to consider (m)
        self.OBSTACLES_SAFETY_DIST = 0.18  # Minimum distance to obstacles (m)
        self.EMERGENCY_STOP_DIST = 0.13  # Distance to trigger emergency stop (m)

        self.SIMULATION_TIME = 2.0  # seconds
        self.TIME_STEP = 0.1  # seconds
        self.HEADING_WEIGHT = 1.0
        self.VELOCITY_WEIGHT = 4.5
        self.OBSTACLE_WEIGHT = 3.1

        #Intermediate feedback variables
        self.steps_counter = 0
        self.feedback_interval = 50  # steps

        # useful lambda functions
        self.checkSafety = lambda lasers: True if np.min(lasers)>self.EMERGENCY_STOP_DIST else False
        self.dist_to_point = lambda robot_pose, point_pose: np.linalg.norm(robot_pose[0:2] - point_pose)



    #####################################
    ##            CALLBACKS            ##
    #####################################

    def odom_callback(self, msg):
        """

        Update robot pose from odometry data.

        """
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        # Yaw extraction from quaternion
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_pose[2] = yaw



    #TODO check if this is correct for real robot
    def landmark_callback(self, msg):
        """

        Update goal pose from landmark data.
        Used in the real robot where the goal is provided as a landmark.

        """
        self.goal_pose[0] = msg.landmarks[0].pose.position.x
        self.goal_pose[1] = msg.landmarks[0].pose.position.y
        #self.get_logger().info(f"Updated Goal Pose: {self.goal_pose}")

    def dynamic_goal_callback(self, msg):
        """

        Update goal pose from dynamic goal topic.
        Used in simulation where the goal is published as an odometry message.

        """
        self.goal_pose[0] = msg.pose.pose.position.x
        self.goal_pose[1] = msg.pose.pose.position.y
        #self.get_logger().info(f"Updated Goal Pose: {self.goal_pose}")


    def laser_callback(self, msg):
        """

        Update laser scan data.
        
        """
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.lasers = self.laser_filter(msg)


    def laser_filter(self, scan_msg):
        """

        Filter laser scan data to reduce noise and limit maximum range, 
        and improve computational efficiency.

        """

        max_range = self.MAX_LASER_RANGE

        lasers = np.array(scan_msg.ranges)
        lasers[lasers > max_range] = max_range
        lasers[np.isinf(lasers)] = max_range
        lasers[np.isnan(lasers)] = 0.0

        # Subdivide lasers in 30 sectors and take the minimum distance in each sector
        sector_size = len(scan_msg.ranges) // self.LASERS_OBS_NUM
        filtered_ranges = []
        for i in range(self.LASERS_OBS_NUM):
            sector_ranges = lasers[i*sector_size : (i+1)*sector_size]
            filtered_ranges.append(min(sector_ranges))
            

        filtered_ranges = np.array(filtered_ranges)


        return filtered_ranges

    
    def stop(self):
        """
        
        Stop the robot by publishing zero velocities.
        Used as emergency stop when an obstacle is too close.

        """
        self.get_logger().info("Stopping the robot.")
        self.cmd_vel_pub.publish(Twist())  # Publish zero velocities to stop the robot



    
    #####################################
    ##          DWA ALGORITHM          ##
    #####################################

    def predict_pose(self, state, u, dt):
        """

        Predict the next pose of the robot based on given state, control inputs, and time step.
        Used to predict robot trajectory during DWA evaluation.  
        Args:
            state: current robot pose [x, y, theta]
            u: control inputs [v, w]
            dt: time step

        """
        next_x = state[0] + u[0] * np.cos(state[2]) * dt
        next_y = state[1] + u[0] * np.sin(state[2]) * dt
        next_th = state[2] + u[1] * dt
        return np.array([next_x, next_y, next_th])


    # Compute the best velocity command
    # This is the core of the DWA algorithm
    def velocity_command(self):
        """

        Test the v and w combinations to avoid obstacles and reach the goal.
        Return the selected velocity command [v, w].

        """

        obstacles = self.get_obstacles()

        v_range = np.linspace(0, self.VMAX, self.V_STEPS)
        w_range = np.linspace(-self.WMAX, self.WMAX, self.W_STEPS)

        # Matrix to store simulated poses for each (v,w) pair in time
        simulated_poses = np.zeros((len(v_range)*len(w_range), int(self.SIMULATION_TIME/self.TIME_STEP)+1,3))
        simulated_poses[:,0, :] = self.robot_pose  # initial pose for all trajectories

        total_scores = np.zeros(len(v_range)*len(w_range))
        min_obstacle_dist = float('inf')

        # Max score initialization
        max_score = -float('inf')

        for ind_v,v in enumerate(v_range):
            for ind_w,w in enumerate(w_range):
                
                u = np.array([v, w])
                min_obstacle_dist = float('inf')

                for ind_t in range(int(self.SIMULATION_TIME/self.TIME_STEP)):
                    # Index of the trajectory being evaluated
                    score_index = ind_v * len(w_range) + ind_w

                    # Predict next pose in the trajectory
                    simulated_poses[score_index, ind_t+1] = self.predict_pose(simulated_poses[score_index,ind_t], u, self.TIME_STEP)
                    
                    # Evaluate trajectory
                    obstacle_score = self.obstacle_score(simulated_poses[score_index, ind_t+1], obstacles) * self.OBSTACLE_WEIGHT
                    
                    if obstacle_score < min_obstacle_dist:
                        min_obstacle_dist = obstacle_score

                    if obstacle_score == 0.0:
                        total_scores[score_index] =  -float('inf')
                        break  # skip this u as it leads to collision
                    
                    if ind_t == int(self.SIMULATION_TIME/self.TIME_STEP)-1:
                        heading_score = self.HEADING_WEIGHT * (np.pi - abs(self.heading_to_goal(simulated_poses[score_index,ind_t+1], self.goal_pose))) # prefer smaller heading angle
                        velocity_score = self.VELOCITY_WEIGHT * v # prefer higher velocities
                        total_scores[score_index] = heading_score+ velocity_score + obstacle_score if total_scores[score_index] == 0 else -float('inf')

        
        best_u_idx = 0
        for ind_score, score in enumerate(total_scores):
            if score > max_score:
                max_score = score
                best_u_idx = ind_score

        v_best_index = best_u_idx // len(w_range)
        w_best_index = best_u_idx % len(w_range)
        u_best = np.array([v_range[v_best_index], w_range[w_best_index]])
        return u_best

    
    def obstacle_score(self, pose, obstacles):
        """

        Check if the given pose collides with any obstacles.
        Return the minimum distance to obstacles or 0.0 if in collision.

        """

        min_dist = self.MAX_LASER_RANGE
        for obs in obstacles:
            dist_to_obs = self.dist_to_point(pose, obs)
            if dist_to_obs < self.OBSTACLES_SAFETY_DIST:
                return 0.0
            if dist_to_obs < min_dist:
                min_dist = dist_to_obs
        return min_dist
    
    def get_obstacles(self):
        """

        Convert laser scan data into obstacle coordinates in the robot's frame.
        Return: np.ndarray of shape (n_obstacles, 2) with (x, y) positions.

        """
        obstacles = []
        angle_increment = (self.angle_max - self.angle_min) / self.LASERS_OBS_NUM
        for i, r in enumerate(self.lasers):
            if r > 0.0 and r < self.MAX_LASER_RANGE:
                x_obs =self.robot_pose[0] + r * np.cos(self.angle_min + i * angle_increment + self.robot_pose[2])
                y_obs =self.robot_pose[1] + r * np.sin(self.angle_min + i * angle_increment + self.robot_pose[2])
                obstacles.append([x_obs, y_obs])
        
        return np.array(obstacles)


    #####################################
    ##        MAIN CONTROLLER         ##
    #####################################
    def go_to_pose(self):
       
        """ Intermediate checkpoint feedback"""
        self.steps_counter += 1
        if self.steps_counter % self.feedback_interval == 0:
            dist = self.dist_to_point(self.robot_pose, self.goal_pose)
            self.goal_feedback_pub.publish(Float32(data=float(dist)))
            self.steps_counter = 0 
       
        """

        Main DWA loop called periodically to compute and publish velocity commands.
        Check for obstacles, goal reaching, and compute best velocity command.

        """
        if self.lasers is None:
            return
        
        if self.checkSafety(self.lasers) == False:
            self.get_logger().warn("Obstacle too close! Stopping robot.")
            self.stop()
            return

        
        dist = self.dist_to_point(self.robot_pose, self.goal_pose)
        if dist < self.GOAL_TOLERANCE:
            self.get_logger().info("Goal Reached!")
            self.stop() 
            return
        
        # Compute best velocity command using DWA
        best_u = self.velocity_command()
        # self.get_logger().info(f"Selected velocities => v: {best_u[0]:.2f} m/s, w: {best_u[1]:.2f} rad/s")

        # Pubblichiamo il risultato del DWA
        cmd = Twist()
        cmd.linear.x = float(best_u[0])
        cmd.angular.z = float(best_u[1])
        self.cmd_vel_pub.publish(cmd)

    def heading_to_goal(self, state, goal):
        """
        
        Compute the heading angle towards the goal pose.

        """
        angle = np.arctan2(
            goal[1] - state[1],
            goal[0] - state[0])
        
        detected_heading = normalize_angle(angle - state[2])
        return detected_heading


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()