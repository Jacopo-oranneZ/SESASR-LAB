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

"""

    QUESTO NODO IMPLEMENTA L'EKF PER IL TASK 2,
    che fonde misure di odometria, IMU e landmark.
    Attingiamo al file Task2.py per le funzioni matematiche 5D.

"""


#--- PARAMETRI ---
PROCESS_NOISE_DIAG = [0.001, 0.001, 0.001, 0.1, 0.1] 
SIGMA_LANDMARK = [0.5, 0.1]
SIGMA_ODOM = [0.05, 0.05]
SIGMA_IMU = [0.02]

class EKFnodeTask2(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf_task2')

        self.landmarks_map = {
            11: (-1.1, -1.1), 12: (-1.1, 0.0), 13: (-1.1, 1.1),
            21: (0.0, -1.1),  22: (0.0, 0.0),  23: (0.0, 1.1),
            31: (1.1, -1.1),  32: (1.1, 0.0),  33: (1.1, 1.1)
        }

        self.ekf = RobotEKF(
            dim_x=5,
            dim_u=0, # Mettiamo 0 perché non usiamo input esterni
            eval_gux=eval_gux_5d,
            eval_Gt=eval_Gt_5d,
            eval_Vt=lambda *args: np.eye(5) 
        )
        
        # Stato Iniziale
        self.ekf.mu = np.array([-2.0, -0.5, 0.0, 0.0, 0.0])
        self.ekf.Mt = np.diag(PROCESS_NOISE_DIAG) # 5x5 Matrix

        # Variabili di supporto
        self.dt = 0.05 # 20 Hz

        # PUBLISHERS
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)

        # SUBSCRIPTIONS
        self.create_subscription(Odometry, '/odom', self.odom_measure_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_measure_callback, 10)
        self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)

        # TIMER PER PREDIZIONE
        self.timer = self.create_timer(self.dt, self.prediction_callback)
        
        self.get_logger().info("EKF Task 2 (Sensor Fusion) Started!")

    ####################################################
    ##    CALLBACKS PER PREDIZIONE E AGGIORNAMENTO    ##
    ####################################################

    def prediction_callback(self):
        """
        
        Esegue il passo di predizione dell'EKF usando il modello di movimento.
        Passiamo u=None perché non usiamo input di controllo esterni.

        """
        dummy_u = None
        
        # Passiamo sigma_u=None per non sovrascrivere Mt con zeri
        self.ekf.predict(u=dummy_u, sigma_u=None, g_extra_args=(self.dt,))
        
        self.publish_ekf_state()

    def odom_measure_callback(self, msg):
        """

        Riceve l'odometria e aggiorna l'EKF.

        """

        # Estraiamo la misura v, w
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
        """

        Riceve la misura IMU e aggiorna l'EKF.

        """

        w_meas = msg.angular_velocity.z
        z = np.array([w_meas])
        Q_imu = np.diag([SIGMA_IMU[0]**2])
        
        # Qui definiamo h_imu inline per comodità, essendo molto semplice
        def h_imu(x, y, theta, v, w): return np.array([w])
        
        # Eseguiamo l'update con la misura IMU
        self.ekf.update(
            z=z, eval_hx=h_imu, eval_Ht=lambda *args: eval_H_imu_5d(),
            Qt=Q_imu, Ht_args=(), hx_args=(*self.ekf.mu,), residual=np.subtract
        )

    def landmark_callback(self, msg):
        Q_land = np.diag([SIGMA_LANDMARK[0]**2, SIGMA_LANDMARK[1]**2])
        for lm in msg.landmarks:
            if lm.id in self.landmarks_map:
                m_x, m_y = self.landmarks_map[lm.id]
                z = np.array([lm.range, lm.bearing])
                self.ekf.update(
                    z=z, eval_hx=self.landmark_model_hx, eval_Ht=eval_H_land_5d,
                    Qt=Q_land, Ht_args=(*self.ekf.mu, m_x, m_y),
                    hx_args=(*self.ekf.mu, m_x, m_y), residual=self.angle_diff
                )



    #########################################
    ##          SUPPORT FUNCTIONS          ##
    #########################################
    def angle_diff(self, z_meas, z_pred):
        """

        Calcola la differenza tra due misure angolari, gestendo il wrapping a [-pi, pi].

        """


        diff = z_meas - z_pred
        diff[1] = math.atan2(math.sin(diff[1]), math.cos(diff[1]))
        return diff


    


    ##########################################
    ##         EKF MODELS FUNCTIONS         ##
    ##########################################

    def landmark_model_hx(self, x, y, theta, v, w, mx, my):
        """
        
        Calcola la misura attesa del landmark dato lo stato e la posizione del landmark.
        
        """
        dx = mx - x; dy = my - y
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - theta
        phi = math.atan2(math.sin(phi), math.cos(phi))
        return np.array([r, phi])


    #########################################
    ##        PUBLISHING FUNCTIONS         ##
    #########################################

    def publish_ekf_state(self):
        """

        Pubblica lo stato stimato dall'EKF come messaggio di Odometry

        """
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
    node = EKFnodeTask2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()