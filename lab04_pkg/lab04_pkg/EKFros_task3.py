import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray
from std_msgs.msg import Header
import numpy as np
import math
import tf_transformations

from lab04_pkg.ekf import RobotEKF
from lab04_pkg.Task2 import (
    eval_gux_5d, eval_Gt_5d, 
    eval_H_land_5d, eval_H_odom_5d, eval_H_imu_5d
)

PROCESS_NOISE_DIAG = [0.001, 0.001, 0.001, 0.1, 0.1] 
SIGMA_LANDMARK = [0.5, 0.1]
SIGMA_ODOM = [0.05, 0.05]
SIGMA_IMU = [0.02]

# Altezza approssimativa della fotocamera del Turtlebot da terra (in metri)
ROBOT_CAMERA_HEIGHT = 0.24 

class EKFnodeTask3(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf_task3')

        # Mappa 3D (x, y, z)
        self.landmarks_map = {
            0: (1.2, 1.68, 0.16), 
            1: (1.68, -0.05, 0.18), 
            2: (3.72, 0.14, 0.22),
            3: (3.75, 1.37, 0.21),  
            4: (2.48, 1.25, 0.22),  
            5: (4.8, 1.87, 0.24),
            
        }

        self.ekf = RobotEKF(
            dim_x=5,
            dim_u=0, 
            eval_gux=eval_gux_5d,
            eval_Gt=eval_Gt_5d,
            eval_Vt=lambda *args: np.eye(5) 
        )
        
        self.ekf.mu = np.array([0.0, 0.77, 0.0, 0.0, 0.0])
        self.ekf.Mt = np.diag(PROCESS_NOISE_DIAG)

        self.dt = 0.05

        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_subscription(Odometry, '/odom', self.odom_measure_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_measure_callback, 10)
        self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10) # Robot reale: /camera/landmarks
        self.timer = self.create_timer(self.dt, self.prediction_callback)
        
        self.get_logger().info("EKF Task 3 (3D Geometry Correction) Started!")

    def prediction_callback(self):
        dummy_u = None
        self.ekf.predict(u=dummy_u, sigma_u=None, g_extra_args=(self.dt,))
        self.publish_ekf_state()

    def odom_measure_callback(self, msg):
        v_meas = msg.twist.twist.linear.x
        w_meas = msg.twist.twist.angular.z
        z = np.array([v_meas, w_meas])
        Q_odom = np.diag([SIGMA_ODOM[0]**2, SIGMA_ODOM[1]**2])
        
        def h_odom(x, y, theta, v, w): return np.array([v, w])
            
        self.ekf.update(
            z=z, eval_hx=h_odom, eval_Ht=lambda *args: eval_H_odom_5d(),
            Qt=Q_odom, Ht_args=(), hx_args=(*self.ekf.mu,), residual=np.subtract
        )

    def imu_measure_callback(self, msg):
        w_meas = msg.angular_velocity.z
        z = np.array([w_meas])
        Q_imu = np.diag([SIGMA_IMU[0]**2])
        
        def h_imu(x, y, theta, v, w): return np.array([w])
            
        self.ekf.update(
            z=z, eval_hx=h_imu, eval_Ht=lambda *args: eval_H_imu_5d(),
            Qt=Q_imu, Ht_args=(), hx_args=(*self.ekf.mu,), residual=np.subtract
        )

    def landmark_callback(self, msg):
        Q_land = np.diag([SIGMA_LANDMARK[0]**2, SIGMA_LANDMARK[1]**2])
        for lm in msg.landmarks:
            if lm.id in self.landmarks_map:
                # Estraiamo anche la Z del landmark!
                m_x, m_y, m_z = self.landmarks_map[lm.id]
                
                z = np.array([lm.range, lm.bearing])
                
                # Passiamo (mx, my, mz) agli argomenti extra
                self.ekf.update(
                    z=z, 
                    eval_hx=self.landmark_model_hx_3d, # Usiamo la nuova funzione 3D
                    eval_Ht=eval_H_land_5d, # Jacobiano (approx 2D va bene, la dz è costante)
                    Qt=Q_land, 
                    Ht_args=(*self.ekf.mu, m_x, m_y), # Per Ht (che è 2D) ci bastano x,y
                    hx_args=(*self.ekf.mu, m_x, m_y, m_z), # Per h(x) servono tutti e 3!
                    residual=self.angle_diff
                )

    def landmark_model_hx_3d(self, x, y, theta, v, w, mx, my, mz):
        """
        Calcola la misura attesa considerando la geometria 3D.
        Range = Ipotenusa 3D tra (x,y,z_robot) e (mx,my,mz_landmark)
        """
        # Differenze planari
        dx = mx - x
        dy = my - y
        
        # Differenza di altezza (Delta Z)
        dz = mz - ROBOT_CAMERA_HEIGHT
        
        # Calcolo Range 3D (Teorema di Pitagora nello spazio)
        r_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Il Bearing rimane planare (azimut)
        phi = math.atan2(dy, dx) - theta
        phi = math.atan2(math.sin(phi), math.cos(phi))
        
        return np.array([r_3d, phi])

    def angle_diff(self, z_meas, z_pred):
        diff = z_meas - z_pred
        diff[1] = math.atan2(math.sin(diff[1]), math.cos(diff[1]))
        return diff

    def publish_ekf_state(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom" 
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = self.ekf.mu[0]
        msg.pose.pose.position.y = self.ekf.mu[1]
        q = tf_transformations.quaternion_from_euler(0, 0, self.ekf.mu[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        msg.twist.twist.linear.x = self.ekf.mu[3]
        msg.twist.twist.angular.z = self.ekf.mu[4]
        self.ekf_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFnodeTask3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()