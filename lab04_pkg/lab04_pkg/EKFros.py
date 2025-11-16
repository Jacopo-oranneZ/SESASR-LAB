from lab04_pkg.EKF import RobotLocalizationEKF
import rclpy
import math
import numpy as np
import tf_transformations

from datetime import datetime
from rclpy.node import Node
from numpy.linalg import inv
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from lab04_pkg.ekf import RobotEKF
from lab04_pkg.TASK0_mot import eval_gux, eval_Gt, eval_Vt
from lab04_pkg.TASK0_sens import eval_Ht, landmarks

class EKFnode(Node):
    def __init__(self):  
        super().__init__('robot_localization_ekf')
        self.ekf = RobotEKF(
            dim_x=3,  # Status dimension [x, y, theta]
            dim_u=2,   # Command dimension [v, w]
            eval_gux=eval_gux,
            eval_Gt=eval_Gt,
            eval_Vt=eval_Vt,
            eval_Ht=eval_Ht,
        )
        # Publishers
        self.odom_publisher = self.create_publisher(Odometry, '/pose', 10)
        self.nav_publisher = self.create_publisher(Odometry, '/ekf', 10)

        # Current time stamp
        msg_nav = Odometry()
        msg_nav.header.stamp = self.get_clock().now().to_msg()
        msg_nav.header.frame_id = 'odom'
        msg_nav.child_frame_id = 'base_link'
    
        # Subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.listener_odom, 10)
        self.landmark_subscriber = self.create_subscription(LandmarkArray, '/landmark', self.listener_landmarks, 10)

        # Timer
        self.timer = self.create_timer(0.05, self.predict)  # 20 Hz
        self.get_logger().info("Robot Localization EKF Node Initialized")

        self.step = 0  # Step counter for the EKF prediction

def main():
    rclpy.init()
    ekf_node = EKFnode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()