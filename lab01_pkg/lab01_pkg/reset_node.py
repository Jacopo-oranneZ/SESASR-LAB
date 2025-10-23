import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool


class ResetNode(Node):
    def __init__(self):
        super().__init__('reset_node')

        self.dm=6.0


        self.subscriber = self.create_subscription(Pose, '/pose', self.pose_callback, 10)
        self.publisher = self.create_publisher(Bool, '/reset', 10)
        


        self.get_logger().info('ResetNode inizializzato.')

    def pose_callback(self, msg: Pose):
        distance = (msg.position.x**2 + msg.position.y**2)**0.5

        if distance >= self.dm:
            reset_msg = Bool()
            reset_msg.data = True
            self.publisher.publish(reset_msg)
            self.get_logger().info('Reset inviato.')




def main(args=None):
    rclpy.init(args=args)
    node = ResetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()