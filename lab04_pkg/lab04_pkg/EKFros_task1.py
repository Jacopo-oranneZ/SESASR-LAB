import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from landmark_msgs.msg import LandmarkArray
from std_msgs.msg import Header
import numpy as np
import math
import tf_transformations

# Importiamo la classe EKF
from lab04_pkg.ekf import RobotEKF

# Importiamo le funzioni matematiche 3D (Task 1) dal Task0
from lab04_pkg.Task0 import (
    eval_gux, eval_Gt, eval_Vt, eval_Ht, landmark_model_hx
)

"""
    QUESTO NODO IMPLEMENTA L'EKF PER IL TASK 1,
    attingendo al file Task0.py per le funzioni matematiche 3D.
"""



# --- PARAMETRI ---
# [alpha1, alpha2, alpha3, alpha4]
# Questi generano la matrice Mt dinamicamente in base alla velocità
A_NOISE = [0.2, 0.05, 0.05, 0.2] 

# [range (m), bearing (rad)]
SIGMA_LANDMARK = [0.5, 0.1] 

class EKFnodeTask1(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf_task1')

        # 1. MAPPA LANDMARK (ID -> x, y)
        self.landmarks_map = {
            11: (-1.1, -1.1), 12: (-1.1, 0.0), 13: (-1.1, 1.1),
            21: (0.0, -1.1),  22: (0.0, 0.0),  23: (0.0, 1.1),
            31: (1.1, -1.1),  32: (1.1, 0.0),  33: (1.1, 1.1)
        }

        # 2. INIZIALIZZAZIONE EKF (Stato 3D)
        self.ekf = RobotEKF(
            dim_x=3, # [x, y, theta]
            dim_u=2, # [v, w]
            eval_gux=eval_gux,
            eval_Gt=eval_Gt,
            eval_Vt=eval_Vt,
            eval_Ht=eval_Ht
        )
        
        # Stato Iniziale
        self.ekf.mu = np.array([-2.0, -0.5, 0.0])
        
        # Matrice Rumore Processo Iniziale (Mt)
        # Sarà sovrascritta dinamicamente nel predict, ma la inizializziamo
        self.ekf.Mt = np.eye(2) 

        # Variabili di supporto per il comando
        self.u = np.array([0.0, 0.0])
        self.sigma_u = np.array([0.001, 0.001]) # Valore piccolo di default

        self.dt = 0.05 # 20 Hz

        # 3. PUBLISHERS
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        
        # 4. SUBSCRIBERS
        
        # Odometria -> INPUT DI CONTROLLO (Serve per il Predict)
        self.create_subscription(Odometry, '/odom', self.odom_input_callback, 10)
        
        # Landmark -> MISURA (Serve per l'Update)
        self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)

        # Timer Predizione -> Esegue il passo di moto
        self.timer = self.create_timer(self.dt, self.prediction_callback)
        
        self.get_logger().info("EKF Task 1 (Standard EKF) Started!")

    ########################################
    ##              CALLBACKS             ##
    ########################################

    def prediction_callback(self):
        """
        
        Esegue il passo di predizione dell'EKF usando l'input di odometria.
        Questo callback è chiamato periodicamente dal timer, ogni dt secondi.

        """        
        self.ekf.predict(u=self.u, sigma_u=self.sigma_u, g_extra_args=(self.dt,))

        self.publish_ekf_state()


    def odom_input_callback(self, msg):
        """

        Riceve l'odometria e aggiorna il comando di controllo e il rumore di processo.
        Questo callback è chiamato ogni volta che arriva un messaggio di odometria.

        """
        # 1. Estraiamo il comando v, w
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.u = np.array([v, w])

        # 2. Calcoliamo il rumore dinamico (Motion Model Probabilistico)
        # sigma^2 = alpha1*v^2 + alpha2*w^2 ...
        sigma_v2 = A_NOISE[0] * v**2 + A_NOISE[1] * w**2
        sigma_w2 = A_NOISE[2] * v**2 + A_NOISE[3] * w**2
        
        # Epsilon per stabilità numerica quando v o w sono zero
        sigma_v2 += 0.0001
        sigma_w2 += 0.0001

        self.sigma_u = np.array([sigma_v2, sigma_w2])


    def landmark_callback(self, msg):
        """

        Riceve le misure dei landmark e aggiorna l'EKF.
        Questo callback è chiamato ogni volta che arriva un messaggio di landmark.

        """

        # Matrice Covarianza Misura Costante
        Q_land = np.diag([SIGMA_LANDMARK[0]**2, SIGMA_LANDMARK[1]**2])

        # Per ogni landmark misurato eseguiamo l'update
        for lm in msg.landmarks:
            lm_id = lm.id
            
            if lm_id in self.landmarks_map:
                m_x, m_y = self.landmarks_map[lm_id]
                z = np.array([lm.range, lm.bearing])

                # Chiamiamo update passando i parametri del landmark
                self.ekf.update(
                    z=z,
                    eval_hx=landmark_model_hx, # Funzione h(x) standard 3D
                    eval_Ht=eval_Ht, # Jacobiano 3D standard
                    Qt=Q_land,
                    Ht_args=(*self.ekf.mu, m_x, m_y),
                    hx_args=(*self.ekf.mu, m_x, m_y),
                    residual=self.angle_diff
                )


    ########################################
    ##          SUPPORT FUNCTIONS         ##
    ########################################

    def angle_diff(self, z_meas, z_pred):
        """

        Calcola la differenza tra due misure angolari, gestendo il wrapping a [-pi, pi].

        """
        diff = z_meas - z_pred
        diff[1] = math.atan2(math.sin(diff[1]), math.cos(diff[1]))
        return diff
    

    
    ##########################################
    ##        PUBLISHING FUNCTIONS         ##
    ##########################################
    def publish_ekf_state(self):
        """
        
        Pubblica lo stato stimato dall'EKF come messaggio di Odometry
        
        """

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom" 
        msg.child_frame_id = "base_link"

        # Posizione
        msg.pose.pose.position.x = self.ekf.mu[0]
        msg.pose.pose.position.y = self.ekf.mu[1]
        msg.pose.pose.position.z = 0.0

        # Orientamento
        q = tf_transformations.quaternion_from_euler(0, 0, self.ekf.mu[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        self.ekf_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFnodeTask1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
