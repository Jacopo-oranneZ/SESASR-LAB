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

# --- CONFIGURAZIONE LABORATORIO ---

# Posizione iniziale (MANTENUTA LA TUA)
INITIAL_POSE = [0.0, 0.77, 0.0] 

# Topic Names
TOPIC_LANDMARKS = '/camera/landmarks'
TOPIC_ODOM = '/odom'
TOPIC_IMU = '/imu'

# Parametri Fisici
ROBOT_CAMERA_HEIGHT = 0.25 # MANTENUTO IL TUO (0.25m)

# --- TUNING DI SICUREZZA (MODIFICATO) ---
# Abbassato a 2.5m per evitare misure rumorose da lontano
MAX_LANDMARK_DIST = 2.5    

# Process Noise (Incertezza Moto)
PROCESS_NOISE_DIAG = [0.001, 0.001, 0.001, 0.1, 0.1] 

# Measurement Noise (Incertezza Sensori)
# AUMENTATO: [1.0m, 0.2rad]. 
# Rende il filtro meno "nervoso" e la traiettoria più liscia.
SIGMA_LANDMARK = [10.0, 5.0] 
SIGMA_ODOM = [0.05, 0.05]
SIGMA_IMU = [0.02]

class EKFnodeTask3(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf_task3')

        # Mappa Landmark 3D
        self.landmarks_map = {
            0: (1.2, 1.68, 0.16), 
            1: (1.68, -0.05, 0.18), 
            2: (3.72, 0.14, 0.22),
            3: (3.75, 1.37, 0.21),  
            4: (2.48, 1.25, 0.22),  
            5: (4.8, 1.87, 0.24),
            6: (2.18, 1.00, 0.24),
            7: (2.94, 2.70, 0.73)
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

        # Gestione Tempo
        self.last_prediction_time = self.get_clock().now()
        self.timer_period = 0.05 

        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        self.create_subscription(Odometry, TOPIC_ODOM, self.odom_measure_callback, 10)
        self.create_subscription(Imu, TOPIC_IMU, self.imu_measure_callback, 10)
        self.create_subscription(LandmarkArray, TOPIC_LANDMARKS, self.landmark_callback, 10)
        
        self.timer = self.create_timer(self.timer_period, self.prediction_callback)
        
        self.get_logger().info(f"EKF Task 3 Ready! Initial Pose: {INITIAL_POSE}")

    def prediction_callback(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_prediction_time).nanoseconds / 1e9
        self.last_prediction_time = current_time
        
        # Clamp per evitare esplosioni se il PC lagga
        if dt > 0.5:
            dt = 0.05 

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
            if lm.id not in self.landmarks_map:
                # Logghiamo solo ogni tanto per non intasare il terminale
                self.get_logger().warn(f"Unknown ID: {lm.id}", throttle_duration_sec=5.0)
                continue

            # 1. Filtro Distanza (Sicurezza base)
            if lm.range > MAX_LANDMARK_DIST:
                continue 

            m_x, m_y, m_z = self.landmarks_map[lm.id]
            z = np.array([lm.range, lm.bearing])
            
            # 2. Controllo Anti-Spuntoni (Gating)
            # Calcoliamo dove dovremmo vedere il landmark secondo la nostra stima attuale
            pred_z = self.landmark_model_hx_3d(*self.ekf.mu, m_x, m_y, m_z)
            
            # Differenza tra misura reale e attesa (solo sul range per semplicità)
            diff_range = abs(z[0] - pred_z[0])
            
            # Se la misura differisce di più di 1.5m dalla stima, è probabilmente un errore
            # o un "fantasma". La scartiamo per proteggere la traiettoria.
            if diff_range > 1.5:
                 self.get_logger().warn(f"Outlier rejected (ID {lm.id}): err={diff_range:.2f}m", throttle_duration_sec=1.0)
                 continue

            # Se passa i controlli, aggiorniamo!
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