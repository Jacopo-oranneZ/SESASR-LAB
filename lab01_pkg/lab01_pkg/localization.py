#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, Quaternion
import math

class Localization(Node):
    def __init__(self):
        super().__init__('localization')

        #Inizializzazione delle variabili di stato
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0         
        self.time = 1.0           
        self.vx_prev = 0.0
        self.vy_prev = 0.0
        self.msg_1 = True  

        #Sottoscrizione al topic /cmd_vel e pubblicazione sul topic /pose
        self.subscriber = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.publisher = self.create_publisher(Pose, '/pose', 10)

        # Log di avvio
        self.get_logger().info('Nodo Localization inizializzato.')



# Callback per la ricezione dei comandi di velocità
    def cmd_callback(self, msg: Twist):
        # Estrazione delle velocità lineari dai messaggi ricevuti
        vx = msg.linear.x
        vy = msg.linear.y

        # Aggiornamento della posizione e dell'orientamento
        self.x += self.vx_prev * self.time
        self.y += self.vy_prev * self.time

        # Calcolo dell'angolo di orientamento theta
        try:
            self.theta = math.atan2(vy, vx)
        except ZeroDivisionError:
            self.theta = 0.0

        # Creazione del quaternion dall'angolo theta
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(self.theta / 2.0)
        q.w = math.cos(self.theta / 2.0)

        # Pubblicazione della posa aggiornata
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = 0.0
        pose.orientation = q

        # Pubblicazione del messaggio di posa
        self.publisher.publish(pose)

        # Aggiornamento delle velocità precedenti
        self.vx_prev = vx
        self.vy_prev = vy

       
        self.get_logger().info(
            f"Pose → x={self.x:.2f}, y={self.y:.2f}, theta={math.degrees(self.theta):.1f}°, "
            f"quat=({q.x:.2f}, {q.y:.2f}, {q.z:.2f}, {q.w:.2f})"
        )

def main(args=None):

    # Inizializzazione del nodo ROS2
    rclpy.init(args=args)
    node = Localization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()