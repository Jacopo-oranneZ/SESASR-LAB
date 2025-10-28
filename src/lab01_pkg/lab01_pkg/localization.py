#!/usr/bin/env python3
import rclpy
import math
from rclpy.node import Node

from geometry_msgs.msg import Pose, Twist, Quaternion


class Localization(Node):
    def __init__(self):
        super().__init__('localization')
        
        #Publisher part
        self.publisher_ = self.create_publisher(Pose, '/pose', 10)
        self.message = Pose()

        #Subscriber part
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.listener_callback, 10)
        


        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_pose)
        self.x=0.0
        self.y=0.0


       

        self.get_logger().info('Nodo localization avviato')



    
    def publish_pose(self):
        self.publisher_.publish(self.message)
        #Log nel bash del messaggio pubblicato
        self.get_logger().info(f'/pose pubblicato: {self.message}')

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


    def listener_callback(self, msg: Twist):
    
        vx = msg.linear.x
        vy = msg.linear.y


        #Velocità comunicate da controller
        self.x += vx
        self.y += vy

        theta = math.atan2(vy, vx) 

        #Parametri di tipo Pose da pubblicare
        msg=Pose()
        msg.position.x=self.x
        msg.position.y=self.y
        msg.position.z=0.0
        msg.orientation=Quaternion(x=0.0, y=0.0, z=math.sin(theta/2), w=math.cos(theta/2)) #Nessuna rotazione implicata

        #Pubblico la posizione attuale
        self.message=msg

        
        

       
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