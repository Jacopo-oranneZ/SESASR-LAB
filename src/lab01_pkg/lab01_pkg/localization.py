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
from rclpy.node import Node

from geometry_msgs.msg import Pose, Twist, Quaternion


class Localization(Node):

    def __init__(self):
        super().__init__('localization')
        
        #Publisher part
        self.publisher_ = self.create_publisher(Twist, '/pose', 10)
        #Subscriber part
        self.subscription = self.create_subscription(Pose, '/cmd_vel', self.listener_callback, 10)
        
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_pose)
        self.x=0.0
        self.y=0.0


       

        self.get_logger().info('Nodo localization avviato')



    
    def publish_pose(self):
        msg = Twist()


      
               
        self.publisher_.publish(msg)




    def listener_callback(self, msg: Twist):
        #Velocità comunicate da controller
        vx=msg.linear.x
        vy=msg.linear.y

        #Parametri di tipo Pose da pubblicare
        msg=Pose()
        self.pose_msg.position.x=self.x
        self.pose_msg.position.y=self.y
        self.pose_msg.position.z=0.0
        self.pose_msg.orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) #Nessuna rotazione implicata

        #Pubblico la posizione attuale
        self.publisher_.publish(self.pose_msg)

        #Log nel bash del messaggio pubblicato
        self.get_logger().info(f'/pose pubblicato: position.x={self.x:1f}, position.y={self.y:1f}, vx={vx:1f}, vy={vy:1f}')
        



def main (args=None):

    rclpy.init(args=args)
    localization = Localization()
    rclpy.spin(localization)
    localization.destroy_node
    rclpy.shutdown()

if __name__=='__main__':
    main()
