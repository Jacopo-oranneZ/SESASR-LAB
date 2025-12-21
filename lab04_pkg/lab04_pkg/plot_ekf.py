import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
import csv
import math
from tf_transformations import euler_from_quaternion

class DataLoggerNode(Node):
    def __init__(self, filename="localization_data_t3.csv"):
        super().__init__('data_logger_node')
        self.filename = filename
        
        self.odom_data = []
        self.ekf_data = []
        self.gt_data = []  # Nuova lista per il Ground Truth

        self.start_time_sec = None
        self.odom_initial_pose = None # Per allineare l'odometria all'origine

        # Sottoscrizioni
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # Assicurati che questo sia il topic corretto nel tuo EKF
        self.ekf_sub = self.create_subscription(Odometry, '/ekf', self.ekf_callback, 10)
        # Sottoscrizione al Ground Truth (VERIFICA IL NOME DEL TOPIC)
        self.gt_sub = self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)
        
        self.get_logger().info(f'DataLogger pronto. Salverà Odom, EKF e Ground Truth su {self.filename}')

    def get_yaw(self, orientation):
        q = (orientation.x, orientation.y, orientation.z, orientation.w)
        return euler_from_quaternion(q)[2]

    def odom_callback(self, msg):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        if self.start_time_sec is None: self.start_time_sec = current_time
        
        # Logica di allineamento all'origine per /odom (Task richiesto)
        if self.odom_initial_pose is None:
            self.odom_initial_pose = {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'yaw': self.get_yaw(msg.pose.pose.orientation)
            }
            x, y, yaw = 0.0, 0.0, 0.0
        else:
            x = msg.pose.pose.position.x - self.odom_initial_pose['x']
            y = msg.pose.pose.position.y - self.odom_initial_pose['y']
            yaw_curr = self.get_yaw(msg.pose.pose.orientation)
            yaw_diff = yaw_curr - self.odom_initial_pose['yaw']
            yaw = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))

        time_rel = current_time - self.start_time_sec
        self.odom_data.append([time_rel, x, y, yaw, 'odom'])

    def ekf_callback(self, msg):
        if self.start_time_sec is None: return
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        time_rel = current_time - self.start_time_sec
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.get_yaw(msg.pose.pose.orientation)
        self.ekf_data.append([time_rel, x, y, yaw, 'ekf'])

    def gt_callback(self, msg):
        # Callback per il Ground Truth
        if self.start_time_sec is None: return
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        time_rel = current_time - self.start_time_sec
        
        # Di solito il GT è assoluto, lo salviamo così com'è
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = self.get_yaw(msg.pose.pose.orientation)
        self.gt_data.append([time_rel, x, y, yaw, 'gt'])

    def save_to_csv(self):
        self.get_logger().info('Salvataggio dati su CSV...')
        all_data = self.odom_data + self.ekf_data + self.gt_data
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time', 'x', 'y', 'yaw', 'source'])
            writer.writerows(all_data)
        self.get_logger().info(f'Fatto. Salvate {len(all_data)} righe.')

    def destroy_node(self):
        self.save_to_csv()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()