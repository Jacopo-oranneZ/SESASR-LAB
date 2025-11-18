import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from landmark_msgs.msg import LandmarkArray
from std_msgs.msg import Header
import numpy as np
import math
import tf_transformations

# Importiamo la tua classe EKF e le funzioni matematiche dal Task0
# Assicurati che i nomi dei file e delle cartelle siano corretti nel tuo workspace
from lab04_pkg.ekf import RobotEKF
from lab04_pkg.Task0 import eval_gux, eval_Gt, eval_Vt, eval_Ht, landmark_range_bearing_sensor

# Parametri di rumore (dal Task 0 o definiti qui)
# a = [alpha1, alpha2, alpha3, alpha4] -> Rumore Moto
A_NOISE = [0.1, 0.01, 0.01, 0.1] 

# Sigma misurazione (range, bearing) -> Rumore Sensore
SIGMA_MEASUREMENT = [0.1, 0.05] # [metri, radianti]

class EKFnode(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf')

        # 1. LA MAPPA DEI LANDMARK (Dalla Tabella 1 del PDF)
        # Dizionario: ID -> (x, y)
        self.landmarks_map = {
            11: (-1.1, -1.1), 12: (-1.1, 0.0), 13: (-1.1, 1.1),
            21: (0.0, -1.1),  22: (0.0, 0.0),  23: (0.0, 1.1),
            31: (1.1, -1.1),  32: (1.1, 0.0),  33: (1.1, 1.1)
        }

        # 2. INIZIALIZZAZIONE EKF
        # Passiamo le funzioni simboliche che hai calcolato nel Task 0
        self.ekf = RobotEKF(
            dim_x=3,
            dim_u=2,
            eval_gux=eval_gux,
            eval_Gt=eval_Gt,
            eval_Vt=eval_Vt,
            eval_Ht=eval_Ht
        )
        
        # Inizializzazione stato (opzionale, parte da 0,0,0)
        self.ekf.mu = np.array([-2.0, -0.5, 0.0])

        # 3. VARIABILI DI SUPPORTO
        self.u = np.array([0.0, 0.0])      # Ultimo comando [v, w]
        self.sigma_u = np.array([0.0, 0.0]) # Rumore corrente
        self.dt = 0.05 # 20 Hz

        # 4. PUB E SUB
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        
        # Sottoscrizione all'odometria (per la PREDIZIONE)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Sottoscrizione ai landmark (per l'UPDATE)
        # Nota: controlla se il topic è /landmark o /landmarks (spesso cambia)
        self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)

        # Timer per la predizione costante a 20Hz
        self.timer = self.create_timer(self.dt, self.prediction_callback)
        
        self.get_logger().info("EKF Node Started correctly!")

    # ---------------------------------------------------------
    # CALLBACK ODOMETRIA (Prepara i dati per la predizione)
    # ---------------------------------------------------------
    def odom_callback(self, msg):
        # Leggiamo v e w
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.u = np.array([v, w])

        # Calcoliamo il rumore dinamico in base alla velocità attuale
        # Modello: sigma^2 = alpha_1 * v^2 + alpha_2 * w^2 ...
        sigma_v2 = A_NOISE[0] * v**2 + A_NOISE[1] * w**2
        sigma_w2 = A_NOISE[2] * v**2 + A_NOISE[3] * w**2
        
        # Salviamo per usarlo nel timer
        self.sigma_u = np.array([sigma_v2, sigma_w2])

    # ---------------------------------------------------------
    # TIMER PREDIZIONE (Esegue il passo di moto ogni 0.05s)
    # ---------------------------------------------------------
    def prediction_callback(self):
        # 1. Eseguiamo la predizione
        # Passiamo dt come tupla in g_extra_args
        self.ekf.predict(u=self.u, sigma_u=self.sigma_u, g_extra_args=(self.dt,))

        # 2. Pubblichiamo il risultato
        self.publish_ekf_state()

    # ---------------------------------------------------------
    # CALLBACK LANDMARK (Esegue l'aggiornamento se vede qualcosa)
    # ---------------------------------------------------------
    def landmark_callback(self, msg):
        # Matrice di covarianza del rumore di misura Q (fissa)
        # Diag(sigma_range^2, sigma_bearing^2)
        Q = np.diag([SIGMA_MEASUREMENT[0]**2, SIGMA_MEASUREMENT[1]**2])

        # Iteriamo su TUTTI i landmark visti in questo frame
        for lm in msg.landmarks:
            lm_id = lm.id
            
            # Controlliamo se conosciamo questo landmark
            if lm_id in self.landmarks_map:
                # Recuperiamo le coordinate note del landmark
                m_x, m_y = self.landmarks_map[lm_id]
                
                # Misura Z ottenuta dal sensore [range, bearing]
                z = np.array([lm.range, lm.bearing])

                # Eseguiamo l'UPDATE
                # hx_args e Ht_args devono contenere lo stato (gestito da EKF) + i parametri extra (mx, my)
                # ATTENZIONE: Le tue funzioni eval_Ht e hx probabilmente si aspettano (x, y, theta, mx, my)
                # Passiamo (mx, my) come argomenti extra
                
                self.ekf.update(
                    z=z,
                    eval_hx=self.landmark_model_hx, # Funzione definita sotto
                    eval_Ht=eval_Ht,                # Jacobiano dal Task 0
                    Qt=Q,
                    Ht_args=(*self.ekf.mu, m_x, m_y), # Argomenti per Ht
                    hx_args=(*self.ekf.mu, m_x, m_y), # Argomenti per hx
                    residual=self.angle_diff          # Funzione per normalizzare gli angoli
                )

    # ---------------------------------------------------------
    # FUNZIONI DI SUPPORTO
    # ---------------------------------------------------------
    
    def landmark_model_hx(self, x, y, theta, mx, my):
        """
        Funzione h(x): calcola la misura ATTESA dato lo stato e il landmark.
        Deve restituire [range_atteso, bearing_atteso]
        """
        dx = mx - x
        dy = my - y
        
        r_expected = math.sqrt(dx**2 + dy**2)
        phi_expected = math.atan2(dy, dx) - theta
        
        # Normalizziamo phi_expected tra -pi e pi
        phi_expected = math.atan2(math.sin(phi_expected), math.cos(phi_expected))
        
        return np.array([r_expected, phi_expected])

    def angle_diff(self, z_meas, z_pred):
        """
        Calcola la differenza z - z_hat gestendo il 'wrap around' dell'angolo.
        z = [range, bearing]
        """
        diff = z_meas - z_pred
        # Normalizziamo SOLO la seconda componente (bearing) che è un angolo
        diff[1] = math.atan2(math.sin(diff[1]), math.cos(diff[1]))
        return diff

    def publish_ekf_state(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom" # O "map", a seconda dell'albero TF
        msg.child_frame_id = "base_link"

        # Posizione
        msg.pose.pose.position.x = self.ekf.mu[0]
        msg.pose.pose.position.y = self.ekf.mu[1]
        msg.pose.pose.position.z = 0.0

        # Orientamento (Da Eulero [rad] a Quaternione)
        q = tf_transformations.quaternion_from_euler(0, 0, self.ekf.mu[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]

        # (Opzionale) Potresti pubblicare anche la covarianza msg.pose.covariance
        # ma richiede di appiattire la matrice 3x3 in un array 6x6
        
        self.ekf_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFnode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()