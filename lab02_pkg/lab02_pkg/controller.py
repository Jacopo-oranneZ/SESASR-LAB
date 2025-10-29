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

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        #Publisher part
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.message = Twist()

        #Subscriber part
        self.create_subscription(LaserScan, "scan", self.listener_scan, qos_profile_sensor_data)
        self.subscription = self.create_subscription(Odometry, '/odom', self.listener_odom, 10)
        self.laser = LaserScan()
        self.odom = Odometry()        
        self.moving_params=Twist()

        #Current phase
        self.phase=0

        self.declare_parameter('linear_velocity', 0.5)
        self.MAX_LINEAR_VELOCITY = self.get_parameter('linear_velocity').get_parameter_value().double_value
        self.declare_parameter('angular_velocity', 0.4)
        self.MAX_ANGULAR_VELOCITY = self.get_parameter('angular_velocity').get_parameter_value().double_value

        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.move)


        self.THRESHOLD=1
        self.theta_threshold = round(90 - math.atan(2*self.THRESHOLD/0.168)*180/math.pi)
        self.turning=False

        self.get_logger().info('Nodo controller avviato')

    def get_yaw(self):
        quaternion = self.odom.pose.pose.orientation
        quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        return yaw % (2*math.pi)


    def stop_turning(self, yaw, threshold=0.5):
        self.get_logger().info(f'Current yaw: {yaw}, Target phase: {self.phase}')
        if (yaw >= self.phase - threshold and yaw<=self.phase + threshold):
            self.get_logger().info('Stopped turning')
            return True
        return False

    
    def move(self):

        if (not self.turning):
            self.moving_params.linear.x=self.MAX_LINEAR_VELOCITY
            self.moving_params.angular.z=0.0

        if(self.turning and self.stop_turning(self.get_yaw())):
            self.moving_params.angular.z=0.0
            self.moving_params.linear.x=self.MAX_LINEAR_VELOCITY
            self.turning=False

        if(self.wall_detector() and not self.turning):
            self.get_logger().info('Wall detected')
            self.turn()
        # self.get_logger().info(f'\# Lasers is {len(self.laser.ranges)}')

        self.publisher_.publish(self.moving_params)


    def turn(self):
        self.turning=True
        if np.mean(self.get_cone(90))> np.mean(self.get_cone(270)):
            self.get_logger().info('Turning left')
            self.moving_params.angular.z=self.MAX_ANGULAR_VELOCITY
            self.phase = (self.phase + (math.pi/2)) % (2*math.pi)
        else:
            self.get_logger().info('Turning right')
            self.moving_params.angular.z=-self.MAX_ANGULAR_VELOCITY
            self.phase = (self.phase + (3*math.pi/2)) % (2*math.pi)
            
        self.moving_params.linear.x=0.0        

    def get_cone(self, center):

        # self.get_logger().info(f'{self.laser.ranges[1]}')
        n=len(self.laser.ranges)

        indices = [(center - self.theta_threshold + i) % n for i in range(2 * self.theta_threshold)]
        result = [self.laser.ranges[i] for i in indices]

        # self.get_logger().info(f'\n\nLASER RANGES\n{self.laser.ranges[center-self.theta_threshold:center]} and {self.laser.ranges[center:center+self.theta_threshold]}\n\n\n')
        
        # self.get_logger().info(
        #     f'LASER CONE center={center} thr={self.theta_threshold} values_sample={result[:8]}')

        return result


    def wall_detector(self):
        
        if self.turning==True: return False
        # self.get_logger().info(f'\n\n\n"{self.get_cone(0)}"\n\n\n')
        if np.mean(self.get_cone(0))< self.THRESHOLD:
            return True
        
        return False



    def listener_scan(self, msg):
        self.laser=msg
        # self.get_logger().info(f'Laser: {self.laser}')
        # Get useful laser data



    def listener_odom(self, msg):
        self.odom=msg
        # self.get_logger().info(f'Odometry: {self.odom}')
        
        
        



def main (args=None):

    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node
    rclpy.shutdown()

if __name__=='__main__':
    main()
