#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Quaternion
import math

class Localization(Node):
    def __init__(self):
        super().__init__('localization')

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0         
        self.time = 1.0           
        self.vx_prev = 0.0
        self.vy_prev = 0.0
        self.msg_1 = True  


        self.subscriber = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.publisher = self.create_publisher(Pose, '/pose', 10)

       
        self.get_logger().info('Nodo Localization inizializzato.')




    def cmd_callback(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y

        if not self.msg_1:
            self.x += self.vx_prev * self.time
            self.y += self.vy_prev * self.time
        else:
            self.msg_1 = False

            self.theta = math.atan2(vy, vx)

        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(self.theta / 2.0)
        q.w = math.cos(self.theta / 2.0)

 
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = 0.0
        pose.orientation = q

       
        self.publisher.publish(pose)

        
        self.vx_prev = vx
        self.vy_prev = vy

       
        self.get_logger().info(
            f"Pose → x={self.x:.2f}, y={self.y:.2f}, theta={math.degrees(self.theta):.1f}°, "
            f"quat=({q.x:.2f}, {q.y:.2f}, {q.z:.2f}, {q.w:.2f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()