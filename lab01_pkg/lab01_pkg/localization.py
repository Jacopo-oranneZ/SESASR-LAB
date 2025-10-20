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

from geometry_msgs.msg import Twist


class Localization(Node):

    def __init__(self):
        super().__init__('localization')
        
        #Publisher part
        self.publisher_ = self.create_publisher(Twist, '/pose', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_pose)
        
        #Subscriber part
        self.subscription = self.create_subscription(
            Twist,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
      

        self.get_logger().info('Localization inizializzato')



    
    def publish_pose(self):
        msg = Twist() #<-- non chiarissima questo comando


      
               
        self.publisher_.publish(msg)




    def listener_callback(self, msg):
        self.get_logger().info('Sto ascoltando: "%s"' % msg.data)



def main (args=None):

    rclpy.init(args=args)
    localization = Localization()
    rclpy.spin(localization)
    localization.destroy_node
    rclpy.shutdown()

if __name__=='__main__':
    main()
