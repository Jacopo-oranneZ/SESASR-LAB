import rclpy
from rclpy.node import Node
from rclpy.time import Time
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

# --- CONFIGURAZIONE LABORATORIO (DA VERIFICARE SUL POSTO) ---

# Posizione iniziale stimata del robot (x, y, theta)
# IMPORTANTE: Metti il robot in questa posizione quando avvii il nodo!
INITIAL_POSE = [0.0, 0.0, 0.0] 

# Topic Names (Verifica con 'ros2 topic list' in lab)
TOPIC_LANDMARKS = '/camera/landmarks' # Spesso è questo su Turtlebot reale
TOPIC_ODOM = '/odom'
TOPIC_IMU = '/imu'

# Parametri Fisici
ROBOT_CAMERA_HEIGHT = 0.25 # Altezza camera da terra (metri)
MAX_LANDMARK_DIST = 4.0    # Ignora landmark più lontani di X metri (riduce rumore)

# Tuning Filtro
PROCESS_NOISE_DIAG = [0.001, 0.001, 0.001, 0.1, 0.1] 
SIGMA_LANDMARK = [0.5, 0.1]
SIGMA_ODOM = [0.05, 0.05]
SIGMA_IMU = [0.02]

class EKFnodeTask3(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf_task3')

        # Mappa Landmark 3D (x, y, z)
        # Se gli ID in lab sono diversi, vedrai un warning nel terminale!
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
        
        # Inizializzazione Stato
        self.ekf.mu = np.array([INITIAL_POSE[0], INITIAL_POSE[1], INITIAL_POSE[2], 0.0, 0.0])
        self.ekf.Mt = np.diag(PROCESS_NOISE_DIAG)

        # Gestione Tempo Robusta
        self.last_prediction_time = self.get_clock().now()
        self.timer_period = 0.05 # Target 20Hz

        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_subscription(Odometry, TOPIC_ODOM, self.odom_measure_callback, 10)
        self.create_subscription(Imu, TOPIC_IMU, self.imu_measure_callback, 10)
        self.create_subscription(LandmarkArray, TOPIC_LANDMARKS, self.landmark_callback, 10)
        
        self.timer = self.create_timer(self.timer_period, self.prediction_callback)
        
        self.get_logger().info(f"EKF Task 3 Started! Listening for landmarks on {TOPIC_LANDMARKS}")

    def prediction_callback(self):
        # Calcolo dt reale per compensare lag della CPU
        current_time = self.get_clock().now()
        dt = (current_time - self.last_prediction_time).nanoseconds / 1e9
        self.last_prediction_time = current_time
        
        # Safety check: se dt è troppo grande (lag enorme), lo cappiamo
        if dt > 0.5:
            self.get_logger().warn(f"Large dt detected: {dt:.4f}s. Lagging?")
            dt = 0.05 # Fallback

        dummy_u = None
        self.ekf.predict(u=dummy_u, sigma_u=None, g_extra_args=(dt,))
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
            # 1. Controllo ID Sconosciuti
            if lm.id not in self.landmarks_map:
                # Throttle per non spammare il log se vede sempre lo stesso tag ignoto
                self.get_logger().warn(f"Unknown Landmark ID detected: {lm.id} - Check Map!", throttle_duration_sec=2.0)
                continue

            # 2. Filtro Distanza (Gating grezzo)
            if lm.range > MAX_LANDMARK_DIST:
                continue # Ignora misure troppo lontane/inaffidabili

            m_x, m_y, m_z = self.landmarks_map[lm.id]
            z = np.array([lm.range, lm.bearing])
            
            self.ekf.update(
                z=z, 
                eval_hx=self.landmark_model_hx_3d, 
                eval_Ht=eval_H_land_5d,
                Qt=Q_land, 
                Ht_args=(*self.ekf.mu, m_x, m_y),
                hx_args=(*self.ekf.mu, m_x, m_y, m_z), 
                residual=self.angle_diff
            )

    def landmark_model_hx_3d(self, x, y, theta, v, w, mx, my, mz):
        # Calcolo distanza 3D (Slant Range)
        dx = mx - x
        dy = my - y
        dz = mz - ROBOT_CAMERA_HEIGHT
        
        r_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        
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