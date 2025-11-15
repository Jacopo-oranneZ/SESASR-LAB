import rclpy
import math
import numpy as np
import tf_transformations

from datetime import datetime
from rclpy.node import Node
from ekf import RobotEKF
from numpy.linalg import inv
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from lab04_pkg.TASK0_mot import eval_gux, eval_Gt, eval_Vt
from lab04_pkg.TASK0_sens import eval_Ht, eval_hx

ekf = RobotEKF(
    dim_x = 3,  # Status dimension [x, y, theta]
    dim_u = 2,   # Command dimension [v, w]
    eval_gux=eval_gux,
    eval_hx=eval_hx,
    eval_Gt=eval_Gt,
    eval_Vt=eval_Vt,
    eval_Ht=eval_Ht,
    )

     # Estimate of the state publisher
    self.ekf_publisher = self.create_publisher(Odometry, '/pose', 10)

    # Subscribers
    self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.listener_odom, 10)

    # Timer
    self.timer = self.create_timer(0.05, self.predict)  # 20 Hz

        self.get_logger().info("Robot Localization EKF Node Initialized")

        self.step = 0  # Step counter for the EKF prediction
       
        
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """

        self.mu = np.zeros((dim_x))  # mean state estimate
        self.Sigma = np.eye(dim_x)  # covariance state estimate
        self.Mt = np.eye(dim_u)  # process noise

        self.eval_gux = eval_gux
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt

        self._I = np.eye(dim_x)  # identity matrix used for computations


