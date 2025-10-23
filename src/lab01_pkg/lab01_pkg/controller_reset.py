# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class ControllerReset(Node):

    def __init__(self):
        super().__init__('controller_reset')

        timer_period = 1.0  # seconds
        
        #DEFINIZIONE DELLE CONDIZIONI DI MOVIMENTO
        self.N = 1 #Tempo di spostamento iniziale (al ciclo n=0) [secondi]
        self.phi = 0 #Valore della fase intesa come fase di spostamento (da 0 a 3 dove 3 è lo spostamento finale lungo y)
        self.t_phi=0
        
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(timer_period, self.publish_vel)
        self.subscriber = self.create_subscription(Bool, '/reset', self.reset_callback, 10)
        self.get_logger().info('Comunicazione con controller avviata.')



    def reset_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info('Reset ricevuto, ripristino stato iniziale.')
            self.N = 1
            self.phi = 0
            self.t_phi = 0
    
    def publish_vel(self):
        msg = Twist() #<-- non chiarissima questo comando
        speed = 1.0 # Impostazione della velocità, fissa a 1 m/s


        if self.phi == 0:
            msg.linear.x = speed
            msg.linear.y = 0.0
        elif self.phi == 1:
           msg.linear.x = 0.0
           msg.linear.y = speed
        elif self.phi == 2:
           msg.linear.x = -speed
           msg.linear.y = 0.0
        elif self.phi == 3:
           msg.linear.x = 0.0
           msg.linear.y = -speed #<- Arrivati a questo pto il ciclo di spostamenti termina


        self.publisher_.publish(msg)
        #Nel seguente comando loggo il messaggio pubblicato nell'attuale callback
        self.get_logger().info(f'/cmd_vel pubblicato: linear.x={msg.linear.x:1f}, linear.y={msg.linear.y:1f},')

        #Iter per l'incrementazione del tempo di spostamento a seconda del numero di callback
        #t_phi e` il trempo trascorso nell'attuale fase
        self.t_phi+=1
        
        #Verifico se e` necessario un incremento temporale
        if self.t_phi >= self.N:
            self.phi = (self.phi + 1) % 4 #Il modulo in questo modo mi restituisce valori da 0 a 3 e in tal modo so` quando posso swhitchare fase
            self.t_phi = 0 #non chiarissimo del perche devo fare quiesto            
            if self.phi == 0:
                self.N +=1
                self.get_logger().info(f'Ciclo completato, incremento del tempo di spostamento a {self.N}')



def main (args=None):

    rclpy.init(args=args)
    controller = ControllerReset()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
