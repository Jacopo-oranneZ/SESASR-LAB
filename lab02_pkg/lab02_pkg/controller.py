# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import math
import numpy as np
from rclpy.node import Node
import tf_transformations

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

# Class definition: definition of a class called Controller, which inherits from Node; 
# Inside it, all publishers, subscribers, parameters, and the control logic will be set up.
class Controller(Node):
    # Creation of the constructor method for the Controller class; ROS2 creates a node named 'controller'.
    def __init__(self):
        super().__init__('controller')

        # Publisher part
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        #$ self.message = Twist()

        # Subscriber part
        self.create_subscription(LaserScan, "scan", self.listener_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.listener_odom, 10)
        self.create_subscription(Odometry, '/ground_truth', self.listener_real, 10)

        # Internal variables
        self.laser = LaserScan()
        self.odom = Odometry()        
        self.real = Odometry()
        self.moving_params=Twist()
        # Current phase. It upgrades and indicates the robot orientation after a turn.
        self.phase=0

        # Parameters part
        self.declare_parameter('linear_velocity', 0.5)
        self.MAX_LINEAR_VELOCITY = self.get_parameter('linear_velocity').get_parameter_value().double_value
        self.declare_parameter('angular_velocity', 0.22)
        self.MAX_ANGULAR_VELOCITY = self.get_parameter('angular_velocity').get_parameter_value().double_value
        # Log the parameters
        self.get_logger().info(f'Max Linear Velocity: {self.MAX_LINEAR_VELOCITY}')
        self.get_logger().info(f'Max Angular Velocity: {self.MAX_ANGULAR_VELOCITY}')

        # Timer part
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.move)

        # Wall detection settings
        self.WALL_THRESHOLD=0.7
        SECURITY_MARGIN=0.01 
        self.ANGLE_THRESHOLD=(math.atan((0.168+SECURITY_MARGIN)/(2*self.WALL_THRESHOLD))*180/math.pi)
        self.angle_increment=0.0 # To be set when the laser data arrives. If 0, laser not ready yet.
        self.turning=False # Flag to indicate if the robot is currently turning
        # Log node start
        self.get_logger().info('Nodo controller avviato')

    #$ def get_yaw(self):
    #$    quaternion = self.odom.pose.pose.orientation
    #$     quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    #$     _, _, yaw = tf_transformations.euler_from_quaternion(quat)
    #$     return yaw % (2*math.pi)

    # Function to determine if the robot should stop turning based on its current yaw and target phase
    def stop_turning(self, yaw, threshold=0.1):
        self.get_logger().info(f'Current yaw: {yaw}, Target phase: {self.phase}')
        if (yaw >= self.phase - threshold and yaw<=self.phase + threshold): 
            self.get_logger().info('Stopped turning')
            return True
        return False

    # Main movement function called by the timer
    def move(self):
        # If the robot is not currently turning, move forward
        if (not self.turning): 
            self.moving_params.linear.x=self.MAX_LINEAR_VELOCITY
            self.moving_params.angular.z=0.0

            # If a wall is detected in front, initiate a turn
            if(self.wall_detector()): 
                self.get_logger().info('Wall detected')
                self.turn()

        # If the robot is currently turning, check if it should stop turning. We don't want 
        # the robot to read info from the scan when it's turning; otherwise it will detect 
        # the wall every 0.1s and never stop turning.
        if(self.turning and self.stop_turning(self.odom[2])): 
            self.moving_params.angular.z=0.0
            self.moving_params.linear.x=self.MAX_LINEAR_VELOCITY
            self.turning=False

        
        # self.get_logger().info(f'\# Lasers is {len(self.laser.ranges)}')
        self.acc_error() # Compute accumulated error between odometry and ground truth
        self.publisher_.publish(self.moving_params) # Publish movement commands

    # Function to initiate a turn based on which side has a larger average laser distance: it determines whether to turn left or right
    def turn(self):
        self.turning=True 
        if np.mean(self.get_cone(90))> np.mean(self.get_cone(270)): # Turn left
            self.get_logger().info('Turning left')
            self.moving_params.angular.z=self.MAX_ANGULAR_VELOCITY
            self.phase = (self.phase + (math.pi/2)) % (2*math.pi) # Update target phase after turning left
        else:   # Turn right
            self.get_logger().info('Turning right')
            self.moving_params.angular.z=-self.MAX_ANGULAR_VELOCITY
            self.phase = (self.phase + (3*math.pi/2)) % (2*math.pi) # Update target phase after turning right
         # Note that every phase is divided by pi/2 to keep it between 0 and 2pi    
        self.moving_params.linear.x=0.0 # Stop linear movement while turning

    # Function to get laser readings within a cone centered at a given angle
    def get_cone(self, center):
        if(self.laser.angle_increment==0): # Laser not ready yet
            self.get_logger().info('Laser not ready yet')
            return []

        # Calculate the number of indices corresponding to the angle threshold. It's necessary because, due to
        # hardware specifications, it is not guaranteed that you will have 360 values in the ranges field. Thus,
        # the index does not correspond directly to the angle in degrees.
        theta_threshold_low=int(self.ANGLE_THRESHOLD // self.angle_increment) 
        theta_threshold_high=int(self.ANGLE_THRESHOLD // self.angle_increment +1) # +1 to include every angle < angle_threshold because index 0 corresponds to an angle greater than 0°
        #$ self.get_logger().info(f'{self.laser.ranges[1]}')
        n=len(self.laser.ranges) # Total number of laser readings, different from 360

        # Special case for front cone (center=0). In this case, we have to consider that the indices wrap around.
        if center==0:
            self.get_logger().info(f'Getting front cone. ANGLE_THRESHOLD={self.ANGLE_THRESHOLD}°, rad: angle_min={self.laser.angle_min}, angle_max={self.laser.angle_max}, angle_increment={self.laser.angle_increment}')
            min_=int((self.ANGLE_THRESHOLD-self.laser.angle_min *(180/math.pi))//self.angle_increment)%360  # number of indices on the left side of the front
            max_=int((self.ANGLE_THRESHOLD-self.laser.angle_max*(180/math.pi))//self.angle_increment)%360   # number of indices on the right side of the front

            self.get_logger().info(f'Getting indices from {center - min_} to {center + max_}')
            indices = [(center - min_ + i) % n for i in range(min_ + max_)]     # indices from center - min_ to center + max_, form left to right
            result = [self.laser.ranges[i] for i in indices]

            return result

        # General case for other cones.
        self.get_logger().info(f'Getting indices from {center-theta_threshold_low} to {center + theta_threshold_high}')
        indices = [(center - theta_threshold_low + i) % n for i in range(theta_threshold_low + theta_threshold_high)]
        result = [self.laser.ranges[i] for i in indices]


        #$ self.get_logger().info(f'\n\nLASER RANGES\n{self.laser.ranges[center-self.theta_threshold:center]} and {self.laser.ranges[center:center+self.theta_threshold]}\n\n\n')
        
        #$ self.get_logger().info(
        #$     f'LASER CONE center={center} thr={self.theta_threshold} values_sample={result[:8]}')

        return result

    # Function to detect if there is a wall in front of the robot
    def wall_detector(self):      
        if self.turning==True: return False     # If the robot is already turning, do not detect walls
        # self.get_logger().info(f'\n\n\n"{self.get_cone(0)}"\n\n\n')
        if np.mean(self.get_cone(0))< self.WALL_THRESHOLD:
            return True
        
        return False

    # Callback function for laser scan data
    def listener_scan(self, msg):
        self.laser=msg
        #$ self.get_logger().info(f'\nMinimum angle:{msg.angle_min}\nAngle increment:{msg.angle_increment*(180/math.pi)}\nMaximum angle:{msg.angle_max}\n')
        self.angle_increment=msg.angle_increment*(180/math.pi)
        #$ self.get_logger().info(f'Laser: {self.laser}')

    # Callback function for odometry data
    def listener_odom(self, msg):
        position = msg.pose.pose.position
        quaternion = msg.pose.pose.orientation
        quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        (_, _, yaw) = tf_transformations.euler_from_quaternion(quat)    # Convert quaternion to Euler angles
        self.odom = (position.x, position.y, yaw % (2*math.pi))     # Store odometry as (x, y, yaw)
       
    # Callback function for ground truth data 
    def listener_real(self, msg):
        position = msg.pose.pose.position
        quaternion = msg.pose.pose.orientation
        quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        (_, _, yaw) = tf_transformations.euler_from_quaternion(quat)    # Convert quaternion to Euler angles
        self.real = (position.x, position.y, yaw % (2*math.pi))    # Store ground truth as (x, y, yaw)

    # Function that computes the accumulated error between odometry and ground truth, providing an indication of odometry accuracy
    def acc_error(self):
        # We compute the value of dx, dy and dtheta
        dx = self.real[0] - self.odom[0]
        dy = self.real[1] - self.odom[1]
        dtheta = self.real[2] - self.odom[2]
        # We compute the distance determined by the components dx and dy
        pos_error = (dx**2 + dy**2)**0.5 
        self.get_logger().info(f'Odometry Error: {pos_error:.3f} m, Yaw Error: {dtheta:.3f} rad')    
       

def main (args=None):

    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node
    rclpy.shutdown()

if __name__=='__main__':
    main()
