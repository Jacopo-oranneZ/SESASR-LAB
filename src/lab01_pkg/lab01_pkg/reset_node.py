import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool


class ResetNode(Node):
    def __init__(self):
        super().__init__('reset_node')

        # Definizione della distanza massima consentita
        self.dm=6.0

        # Sottoscrizione al topic /pose e pubblicazione sul topic /reset
        self.subscriber = self.create_subscription(Pose, '/pose', self.pose_callback, 10)
        self.publisher = self.create_publisher(Bool, '/reset', 10)
        


        self.get_logger().info('ResetNode inizializzato.')

    def pose_callback(self, msg: Pose):
        # Calcolo della distanza dalla posizione iniziale
        distance = (msg.position.x**2 + msg.position.y**2)**0.5

        # Creazione e pubblicazione del messaggio di reset
        reset_msg = Bool()

        # Se la distanza supera la soglia, setta data a True, altrimenti False
        reset_msg.data = True if distance >= self.dm else False
            
        # Effettivo invio del messaggio di reset
        self.get_logger().info(f'Reset: {reset_msg.data}.')
        self.publisher.publish(reset_msg)



def main(args=None):

    # Inizializzazione del nodo ROS2
    rclpy.init(args=args)
    node = ResetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()