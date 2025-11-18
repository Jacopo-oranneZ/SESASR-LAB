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
from lab04_pkg.ekf import RobotEKF, Mt
from lab04_pkg.Task0 import eval_gux, eval_Gt, eval_Vt, eval_Ht, a


# TODO: check sigma_u definition


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
        self.dt = 0.05  # Time step for the EKF prediction
        self.timer = self.create_timer(self.dt, self.predict)  # 20 Hz
        self.get_logger().info("Robot Localization EKF Node Initialized")

        # self.step = 0  # Step counter for the EKF prediction

    def listener_odom(self, msg):
        # Extract linear and angular velocities from the odometry message
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z
        self.u = np.array([self.v, self.w])

        # Define standard deviations for the control inputs
        self.sigma_u = np.array([a[0] * self.v**2 + a[1] * self.w**2, a[2] * self.v**2 + a[3] * self.w**2])

    def predict(self, u, sigma_u, g_extra_args=()):


        self.mu = self.eval_gux(self.mu, u, sigma_u, *g_extra_args)

        args = (*self.mu, *u)
        # Update the covariance matrix of the state prediction,
        # you need to evaluate the Jacobians Gt and Vt

        Gt = self.eval_Gt(*args, *g_extra_args)
        Vt = self.eval_Vt(*args, *g_extra_args)
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

        
        
   

    

def main():
    rclpy.init()
    ekf_node = EKFnode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()