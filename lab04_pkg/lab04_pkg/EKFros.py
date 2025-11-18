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
from lab04_pkg.Task0 import eval_gux, eval_Gt, eval_Vt, eval_Ht, sigma, eval_hx, a


# TODO: check sigma_u definition
# TODO: gestire caso w=0 in task0
# TODO: controllare eval_hx e eval_Ht

# Funzione per gestire la differenza tra angoli (wrap-around)
def residual_func(z, z_hat):
    diff = z - z_hat
    diff[1] = (diff[1] + math.pi) % (2 * math.pi) - math.pi
    return diff
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
            sigma = sigma,
            a = a
        )

        # Mappa Landmark (ID -> coordinate x, y)
        self.landmarks_map = {
            11: (-1.1, -1.1), 12: (-1.1, 0.0), 13: (-1.1, 1.1),
            21: (0.0, -1.1),  22: (0.0, 0.0),  23: (0.0, 1.1),
            31: (1.1, -1.1),  32: (1.1, 0.0),  33: (1.1, 1.1)
        }
        
        # Matrice Qt (Rumore Sensore - COSTANTE)
        # Range: 0.3m, Bearing: pi/24 rad
        self.Qt = np.diag([0.3**2, (math.pi/24)**2]) 
        
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
        self.landmark_subscriber = self.create_subscription(LandmarkArray, '/landmark', self.listener_landmark, 10)

        # Timer
        self.dt = 0.05  # Time step for the EKF prediction
        self.timer = self.create_timer(self.dt, self.prediction_callback)  # 20 Hz
        self.get_logger().info("Robot Localization EKF Node Initialized")

        # self.step = 0  # Step counter for the EKF prediction

    def listener_odom(self, msg):
        # Extract linear and angular velocities from the odometry message
        self.v = msg.twist.twist.linear.x
        self.w = msg.twist.twist.angular.z
        self.u = np.array([self.v, self.w])

        self.sigma_u = np.array([math.sqrt(a[0]*self.u[0]**2 + a[1]*self.u[1]**2), math.sqrt(a[2]*self.u[0]**2 + a[3]*self.u[1]**2)])  # Example noise model


    def prediction_callback(self):
        # Aggiorna la matrice di rumore di processo Mt nella classe EKF
        self.ekf.Mt = np.diag(self.sigma_u**2)
        # self.ekf.predict(u=self.u, sigma_u=noise, g_extra_args=(self.dt,))
        self.ekf.predict(self.u, self.sigma_u, g_extra_args=(self.dt,))

    def listener_landmark(self, msg):
        eval_hx = self.landmark_model
        for lmk in msg.landmarks:
            lid = lmk.id
            
            if lid in self.landmarks_map:
                mx, my = self.landmarks_map[lid]
                z = np.array([lmk.range, lmk.bearing])
                
                # Argomenti extra: (x, y, theta, mx, my)
                Hx_args = (*self.ekf.mu, mx, my)
                hx_args = (*self.ekf.mu, mx, my)

                self.ekf.update(
                    z=z, 
                    eval_hx=eval_hx, # Usiamo quella importata
                    eval_Ht=eval_Ht, 
                    Qt=self.Qt, 
                    Ht_args=Hx_args, 
                    hx_args=hx_args, 
                    residual=residual_func
                )
    
    def publish_ekf(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        
        msg.pose.pose.position.x = self.ekf.mu[0]
        msg.pose.pose.position.y = self.ekf.mu[1]
        
        theta = self.ekf.mu[2]
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        
        self.nav_publisher.publish(msg)

def main():
    rclpy.init()
    ekf_node = EKFnode()
    rclpy.spin(ekf_node)
    ekf_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()