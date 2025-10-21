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
from rclpy.node import Node

from geometry_msgs.msg import Laserscan, Odometry


class Controller(Node):

    def __init__(self):
        super().__init__('controller')
        
        #Publisher part
        self.publisher_ = self.create_publisher(Odometry, '/odom', 10)
        self.message = Odometry()

        #Subscriber part
        self.subscription = self.create_subscription(Laserscan, '/scan', self.listener_callback, 10)
        


        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_pose)


       

        self.get_logger().info('Nodo controller avviato')



    
    def publish_pose(self):
        # self.publisher_.publish(self.message)
        #Log nel bash del messaggio pubblicato
        self.get_logger().info(f'/pose pubblicato: {self.message}')





    def listener_callback(self, msg):
    
        self.message=msg

        
        



def main (args=None):

    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node
    rclpy.shutdown()

if __name__=='__main__':
    main()
