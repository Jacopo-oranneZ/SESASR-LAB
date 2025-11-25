import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from landmark_msgs.msg import LandmarkArray
from std_msgs.msg import Header
import numpy as np
import math
import tf_transformations

# EKF class and Task 0's function are imported 
from lab04_pkg.ekf import RobotEKF

# Importing necessary functions from Task0
from lab04_pkg.Task0 import (
    eval_gux, eval_Gt, eval_Vt, eval_Ht
)

# -----------------------------------
#       PARAMETERS 
# -----------------------------------

# Defining noise's parameters (from Task 0 or defined here)
# a = [alpha1, alpha2, alpha3, alpha4] -> Motion Noise
A_NOISE = [0.2, 0.05, 0.05, 0.2] 
# Measurement noise parameters (Landmark)
SIGMA_LANDMARK = [0.5, 0.1] # [range (m), bearing (rad)]

#-----------------------------------
#       EKF NODE - TASK 1
#-----------------------------------

class EKFnodeTask1(Node):
    def __init__(self):
        super().__init__('robot_localization_ekf_task1')

        # 1. THE LANDMARK MAP (informations can be changed)
        # as dictionary: ID -> (x, y)
        self.landmarks_map = {
            11: (-1.1, -1.1), 12: (-1.1, 0.0), 13: (-1.1, 1.1),
            21: (0.0, -1.1),  22: (0.0, 0.0),  23: (0.0, 1.1),
            31: (1.1, -1.1),  32: (1.1, 0.0),  33: (1.1, 1.1)
        }

        # 2. EKF INITIALIZATION
        # Passing the symbolic functions calculated in Task 0 and defining the dimensions of the state and control
        # In this case dim_x=3 (x, y, theta) and dim_u=2 (v, w)
        self.ekf = RobotEKF(
            dim_x=3, # [x, y, theta]
            dim_u=2, # [v, w] 
            eval_gux=eval_gux,
            eval_Gt=eval_Gt,
            eval_Vt=eval_Vt,
            eval_Ht=eval_Ht
        )
        
        # Initialization of the state (the closer to the real one, the better)
        self.ekf.mu = np.array([-2.0, -0.5, 0.0])
        
        # Initial Covariance Matrix
        # It will be dynamically overwritten in the predict step, but we initialize it here
        self.ekf.Mt = np.eye(2) 

        # Initialization of the control input and noise
        self.u = np.array([0.0, 0.0])  # Last command [v, w]
        self.sigma_u = np.array([0.001, 0.001]) # # Current noise - noise parameters of the motion model [a1, a2, a3, a4, a5, a6] expressed as [std_dev_v, std_dev_w]

        # Definition of the time step - we want to perform the prediction step at a fixed rate of 20 Hz
        self.dt = 0.05 # 20 Hz

         # 3. PUB AND SUBSCRIBE
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        # Subscription to odometry (for PREDICTION)
        self.create_subscription(Odometry, '/odom', self.odom_input_callback, 10)
        # Subscription to landmarks (for UPDATE)
        self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        # Timer for constant prediction at 20Hz -> Executes the motion step
        self.timer = self.create_timer(self.dt, self.prediction_callback)
        
        self.get_logger().info("EKF Task 1 (Standard EKF) Started!")

    # ---------------------------------------------------------
    #           PREDICTION (Timer 20Hz)
    # ---------------------------------------------------------
    def prediction_callback(self):
        # In Task 1, we use odometry (self.u) as control input
        # and self.sigma_u to calculate the motion uncertainty.
        self.ekf.predict(u=self.u, sigma_u=self.sigma_u, g_extra_args=(self.dt,))
        # Publish the result
        self.publish_ekf_state()

    # ---------------------------------------------------------
    #       INPUT: ODOMETRY (Calculates u and sigma_u)
    # ---------------------------------------------------------
    def odom_input_callback(self, msg):
        # Reads v and w
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.u = np.array([v, w])

        # Calculate dynamic noise based on current velocity
        # sigma^2 = alpha_1 * v^2 + alpha_2 * w^2 ...
        sigma_v2 = A_NOISE[0] * v**2 + A_NOISE[1] * w**2
        sigma_w2 = A_NOISE[2] * v**2 + A_NOISE[3] * w**2
        
        # Epsilon for numerical stability
        sigma_v2 += 0.0001
        sigma_w2 += 0.0001

        self.sigma_u = np.array([sigma_v2, sigma_w2]) # Saving std deviations

    # ---------------------------------------------------------
    #        UPDATE: LANDMARKS (Measurement r, phi)
    # ---------------------------------------------------------
    def landmark_callback(self, msg):
        # Measurement noise covariance matrix Q (fixed) 
        Q_land = np.diag([SIGMA_LANDMARK[0]**2, SIGMA_LANDMARK[1]**2]) # Diag(sigma_range^2, sigma_bearing^2)

        # Iterate over ALL landmarks seen in this frame
        for lm in msg.landmarks:
            lm_id = lm.id
            
            # Check if we know this landmark
            if lm_id in self.landmarks_map:
                # Retrieve known coordinates of the landmark
                m_x, m_y = self.landmarks_map[lm_id]
                 # Measurement Z obtained from the sensor [range, bearing]
                z = np.array([lm.range, lm.bearing])

                # Perform the UPDATE  
                self.ekf.update(
                    z=z,
                    eval_hx=self.landmark_model_hx, # Function defined below
                    eval_Ht=eval_Ht, # Jacobian from Task 0
                    Qt=Q_land,
                    Ht_args=(*self.ekf.mu, m_x, m_y), # Arguments for Ht
                    hx_args=(*self.ekf.mu, m_x, m_y), # Arguments for hx
                    residual=self.angle_diff         # Function to normalize angles
                )

    # ---------------------------------------------------------
    #            UTILS
    # ---------------------------------------------------------
    def landmark_model_hx(self, x, y, theta, mx, my):
        # Function h(x): calculates the expected measurement given the state and the landmark.
        # It should return [expected_range, expected_bearing]
        
        dx = mx - x
        dy = my - y
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - theta
        # Normalize phi_expected between -pi and pi
        phi = math.atan2(math.sin(phi), math.cos(phi))
        return np.array([r, phi])

    def angle_diff(self, z_meas, z_pred):
        # Calculates the difference z - z_hat handling the angle 'wrap around'.
        # z = [range, bearing]
        diff = z_meas - z_pred
        # Normalize only the second component (bearing) which is an angle
        diff[1] = math.atan2(math.sin(diff[1]), math.cos(diff[1]))
        return diff

    def publish_ekf_state(self):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"  # Or "map", depending on the TF tree
        msg.child_frame_id = "base_link"

        # Position
        msg.pose.pose.position.x = self.ekf.mu[0]
        msg.pose.pose.position.y = self.ekf.mu[1]
        msg.pose.pose.position.z = 0.0

        # Orientation
        q = tf_transformations.quaternion_from_euler(0, 0, self.ekf.mu[2])
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        
        # Note: In Task 1 we do not estimate velocities (they are inputs), 
        # so the twist remains empty or we copy that of odom if we want.

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
    
