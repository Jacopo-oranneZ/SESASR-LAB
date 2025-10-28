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

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        #Publisher part
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.message = Twist()

        #Subscriber part
        self.subscription = self.create_subscription(LaserScan, '/scan', self.listener_scan, 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.listener_odom, 10)
        self.laser = LaserScan()
        self.odom = Odometry()        
        self.moving_params=Twist()




        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.move)


        self.THRESHOLD=1
        self.theta_threshold = round(90 - math.atan(2*self.THRESHOLD/0.168)*180/math.pi)

        self.get_logger().info('Nodo controller avviato')



    
    def move(self):
        
        self.moving_params.angular.z=0.0
        self.moving_params.linear.x=0.5


        if(self.wall_detector()):
            self.get_logger().info('Wall detected')
            self.turn()
        # self.get_logger().info(f'\# Lasers is {len(self.laser.ranges)}')

        self.publisher_.publish(self.moving_params)


    def turn(self):

        if np.mean(self.get_cone(90))> np.mean(self.get_cone(270)):
            self.moving_params.angular.z=0.5
            self.moving_params.linear.x=0.0
            self.publisher_.publish(self.moving_params)
        else:
            self.moving_params.angular.z=-0.5
            self.moving_params.linear.x=0.0
            self.publisher_.publish(self.moving_params)
        

    def get_cone(self, center):

        # self.get_logger().info(f'{self.laser.ranges[1]}')
        result = []
        result.append(self.laser.ranges[center-self.theta_threshold:])
        result.append(self.laser.ranges[center:center+self.theta_threshold])

        
        for i in range(len(result)):
            if result[i]=='inf':
                result[i] = 3.5
        #TODO Da sistemare per i primi gradi
        self.get_logger().info(f'\n\n{result}\n\n')

        return result


    def wall_detector(self):

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
