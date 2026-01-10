import numpy as np
import rclpy
from rclpy.node import Node
from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String

# Imports per il salvataggio dati
import os
import csv
from datetime import datetime

# Funzione helper definita qui per rendere il codice autosufficiente
def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        super().__init__('Obstacle_Avoidance_Node')

        # Timer for controller callback
        self.timer_period = 1/15  # 15 Hz
        self.timer = self.create_timer(self.timer_period, self.go_to_pose)
        
        # --- SUBSCRIPTIONS ---
        self.create_subscription(LandmarkArray, '/camera/landmarks', self.landmark_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Simulation subscription (lasciato per compatibilità)
        # self.create_subscription(Odometry, '/dynamic_goal_pose', self.dynamic_goal_callback, 10)

        # --- PUBLISHERS ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.feedback_pub = self.create_publisher(String, '/navigation_feedback', 10)
        self.distance_pub = self.create_publisher(Float32, '/distance_to_goal', 10)

        # --- DATA LOGGING SETUP ---
        # Creiamo una cartella per i log se non esiste
        self.log_dir = os.path.expanduser('~/dwa_logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Nome file unico basato sull'orario
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.log_dir, f'task3_data_{timestamp_str}.csv')
        self.txt_filename = os.path.join(self.log_dir, f'task3_report_{timestamp_str}.txt')
        
        self.get_logger().info(f"Logging data to: {self.csv_filename}")

        # OTTIMIZZAZIONE I/O: Apriamo il file qui e lo teniamo aperto
        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Scriviamo l'header
        self.csv_writer.writerow(['timestamp', 'dist_to_goal', 'bearing_error', 'min_obstacle_dist', 'v_cmd', 'w_cmd', 'is_tracking', 'is_visible'])

        # General variables
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.goal_pose = np.array([0.0, 0.0])
        self.lasers = None
        self.angle_min = 0
        self.angle_max = 0
        self.angle_increment = 0
        
        # Variabili di stato tracking
        self.goal_received = False
        self.last_landmark_time = 0.0 # Per gestire il timeout visivo
        self.VISIBILITY_TIMEOUT = 1.0 # Secondi dopo i quali il target è considerato perso

        # --- TUNING PER ROBOT REALE ---
        self.OPTIMAL_DIST = 0.22 # Distanza target ottimale
        self.LASERS_OBS_NUM = 30 
        
        # SAFETY NOTE: Turtlebot3 Burger max speed ~0.22 m/s.
        self.VMAX = 0.08
        self.WMAX = 0.4  
        
        self.V_STEPS = 15 
        self.W_STEPS = 35 
        self.MAX_LASER_RANGE = 3.5 
        self.OBSTACLES_SAFETY_DIST = 0.18 
        self.EMERGENCY_STOP_DIST = 0.12 
        self.SLOW_DOWN_DIST = 0.5
        self.VISIBILITY_THRESHOLD = 0.3
        self.CAMERA_OFFSET_X = 0.05 

        # --- METRICS VARIABLES ---
        self.METRICS_UPDATE_RATE = 50 
        
        self.total_steps = 0
        self.tracked_steps = 0 
        self.collision_count = 0
        
        # RMSE Accumulators
        self.sq_error_dist = 0.0
        self.sq_error_bearing = 0.0
        
        # Lidar Stats
        self.global_min_obstacle_dist = float('inf')
        self.cumulative_avg_lidar_dist = 0.0
        self.last_print_time = self.get_clock().now().nanoseconds / 1e9 # Init variable
        
        self.TRACKING_MAX_DIST = 2.0 

        # Parameters for Simulation/Prediction
        self.SIMULATION_TIME = 2.0
        self.TIME_STEP = 0.1 
        
        # DWA Weights
        self.HEADING_WEIGHT = 1.2
        self.VELOCITY_WEIGHT = 4.0
        self.OBSTACLE_WEIGHT = 1.5
        self.VELOCITY_REDUCTION_WEIGHT = 0.5
        self.VISIBILITY_WEIGHT = 3.0

        # Timeout logic (Opzionale/Disabilitato ma variabile presente per sicurezza)
        self.MAX_STEPS_TIMEOUT = 15 * 60 * 3 

        # Lambda utils
        self.checkSafety = lambda lasers: True if np.min(lasers) > self.EMERGENCY_STOP_DIST else False
        self.dist_to_point = lambda robot_pose, point_pose: np.linalg.norm(robot_pose[0:2] - point_pose)


    #####################################
    ##            CALLBACKS            ##
    #####################################

    def odom_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_pose[2] = yaw

    def landmark_callback(self, msg):
        if not msg.landmarks: 
            self.get_logger().warn("Nessun landmark rilevato")
            return 

        # FIX: Aggiorniamo timestamp e flag. Non incrementiamo metriche qui (si fa nel loop)
        self.goal_received = True
        self.last_landmark_time = self.get_clock().now().nanoseconds / 1e9

        try:
            r = msg.landmarks[0].range
            b = msg.landmarks[0].bearing
        except AttributeError:
            # Fallback debug
            self.get_logger().error("Landmark message missing range/bearing attributes.")
            return

        # Conversione Polare -> Cartesiana (Frame Robot)
        rel_x = r * np.cos(b)
        rel_y = r * np.sin(b)

        # Aggiungiamo l'offset della camera
        rel_x += self.CAMERA_OFFSET_X 

        # Trasformazione: Robot Frame -> Odom Frame
        theta = self.robot_pose[2]
        glob_x = self.robot_pose[0] + (rel_x * np.cos(theta) - rel_y * np.sin(theta))
        glob_y = self.robot_pose[1] + (rel_x * np.sin(theta) + rel_y * np.cos(theta))

        self.goal_pose[0] = glob_x
        self.goal_pose[1] = glob_y
        self.get_logger().info(f"Tag visto: Range={r:.2f}, Bearing={b:.2f}")


    def dynamic_goal_callback(self, msg):
        self.goal_received = True
        self.last_landmark_time = self.get_clock().now().nanoseconds / 1e9
        self.goal_pose[0] = msg.pose.pose.position.x
        self.goal_pose[1] = msg.pose.pose.position.y

    def laser_callback(self, msg):
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.lasers = self.laser_filter(msg)

    def laser_filter(self, scan_msg):
        max_range = self.MAX_LASER_RANGE
        lasers = np.array(scan_msg.ranges)
        lasers[np.isnan(lasers)] = max_range 
        lasers[np.isinf(lasers)] = max_range
        lasers[lasers > max_range] = max_range
        lasers[lasers < 0.05] = max_range 

        sector_size = len(scan_msg.ranges) // self.LASERS_OBS_NUM
        filtered_ranges = []
        for i in range(self.LASERS_OBS_NUM):
            start_idx = i * sector_size
            end_idx = (i + 1) * sector_size
            if start_idx >= len(lasers): break
            
            sector_ranges = lasers[start_idx : end_idx]
            if len(sector_ranges) > 0:
                filtered_ranges.append(np.min(sector_ranges))
            else:
                filtered_ranges.append(max_range)

        return np.array(filtered_ranges)

    def stop(self):
        self.get_logger().info("Stopping the robot.")
        self.cmd_vel_pub.publish(Twist())

    #####################################
    ##          DWA ALGORITHM          ##
    #####################################

    def predict_pose(self, state, u, dt):
        next_x = state[0] + u[0] * np.cos(state[2]) * dt
        next_y = state[1] + u[0] * np.sin(state[2]) * dt
        next_th = state[2] + u[1] * dt
        return np.array([next_x, next_y, next_th])

    def velocity_command(self):
        obstacles = self.get_obstacles()
        v_range = np.linspace(0, self.VMAX, self.V_STEPS)
        w_range = np.linspace(-self.WMAX, self.WMAX, self.W_STEPS)

        simulated_poses = np.zeros((self.V_STEPS*self.W_STEPS, int(self.SIMULATION_TIME/self.TIME_STEP)+1, 3))
        simulated_poses[:,0, :] = self.robot_pose

        total_scores = np.full(self.V_STEPS*self.W_STEPS, -float('inf'))

        for ind_v, v in enumerate(v_range):
            for ind_w, w in enumerate(w_range):
                u = np.array([v, w])
                score_index = ind_v * self.W_STEPS + ind_w
                min_obstacle_dist = float('inf')
                collision_detected = False

                for ind_t in range(int(self.SIMULATION_TIME/self.TIME_STEP)):
                    simulated_poses[score_index, ind_t+1] = self.predict_pose(simulated_poses[score_index, ind_t], u, self.TIME_STEP)
                    dist_obs = self.obstacle_score(simulated_poses[score_index, ind_t+1], obstacles)
                    if dist_obs == 0.0:
                        collision_detected = True
                        break 
                    if dist_obs < min_obstacle_dist:
                        min_obstacle_dist = dist_obs

                if collision_detected:
                    total_scores[score_index] = -float('inf')
                    continue

                final_pose = simulated_poses[score_index, -1]
                
                head_score = self.HEADING_WEIGHT * (np.pi - abs(self.heading_to_goal(final_pose, self.goal_pose)))
                vel_score = self.VELOCITY_WEIGHT * v
                
                dist_from_goal = self.dist_to_point(final_pose, self.goal_pose)
                vel_reduction = 0.0
                if dist_from_goal < self.SLOW_DOWN_DIST:
                    vel_reduction = self.VELOCITY_REDUCTION_WEIGHT * (self.SLOW_DOWN_DIST - dist_from_goal)

                obs_score = self.OBSTACLE_WEIGHT * min_obstacle_dist
                vis_score = self.VISIBILITY_WEIGHT * self.get_visibility(final_pose, obstacles)
                total_scores[score_index] = head_score + vel_score + obs_score - vel_reduction + vis_score

        best_u_idx = np.argmax(total_scores)
        max_score = total_scores[best_u_idx]

        if max_score == -float('inf'):
            self.get_logger().warn("DWA: No valid path found! Recovery.")
            return np.array([0.0, 0.3])

        v_best_index = best_u_idx // self.W_STEPS
        w_best_index = best_u_idx % self.W_STEPS
        self.get_logger().info(f" Head:{head_score:.2f} Vel:{vel_score:.2f} Obs:{obs_score:.2f} Vis:{vis_score:.2f} Reduct:{vel_reduction:.2f}")

        return np.array([v_range[v_best_index], w_range[w_best_index]])

    def get_visibility(self, pose, obstacles):
        min_dist = float('inf')
        robottogoal = (self.goal_pose - pose[:2])
        goal_dist = np.linalg.norm(robottogoal)
        if goal_dist < 0.01: return 1.0

        for obs in obstacles:
            posetoobs = (obs - pose[:2])
            projection = np.dot(posetoobs, robottogoal) / goal_dist
            
            if 0 <= projection <= goal_dist:
                dist = np.sqrt(abs(np.linalg.norm(posetoobs)**2 - projection**2))
                if dist < min_dist: min_dist = dist

        result = min_dist / self.VISIBILITY_THRESHOLD
        return min(result, 1.0)

    def obstacle_score(self, pose, obstacles):
        min_dist = self.MAX_LASER_RANGE
        for obs in obstacles:
            dist_to_obs = self.dist_to_point(pose, obs)
            if dist_to_obs < self.OBSTACLES_SAFETY_DIST:
                return 0.0
            if dist_to_obs < min_dist:
                min_dist = dist_to_obs
        return min_dist

    def get_obstacles(self):
        obstacles = []
        if self.LASERS_OBS_NUM == 0: return np.array([])
        effective_increment = (self.angle_max - self.angle_min) / self.LASERS_OBS_NUM
        
        for i, r in enumerate(self.lasers):
            if r > 0.0 and r < self.MAX_LASER_RANGE:
                angle = self.angle_min + (i * effective_increment) + (effective_increment/2) + self.robot_pose[2]
                x_obs = self.robot_pose[0] + r * np.cos(angle)
                y_obs = self.robot_pose[1] + r * np.sin(angle)
                obstacles.append([x_obs, y_obs])
        
        if not obstacles: return np.array([])
        return np.array(obstacles)

    def heading_to_goal(self, state, goal):
        angle = np.arctan2(goal[1] - state[1], goal[0] - state[0])
        return normalize_angle(angle - state[2])

    #####################################
    ##          MAIN CONTROLLER        ##
    #####################################

    def go_to_pose(self):
        if self.lasers is None: return
        
        if not self.checkSafety(self.lasers):
            self.get_logger().warn("EMERGENCY STOP")
            self.stop()
            self.collision_count += 1
            return

        if not self.goal_received: return

        self.total_steps += 1
        # Timeout rimosso come richiesto


        

        # --- FIX TRACKING: Check visibilità ---
        current_time = self.get_clock().now().nanoseconds / 1e9
        target_visible = False
        if self.last_landmark_time > 0:
            time_since_seen = current_time - self.last_landmark_time
            if time_since_seen < self.VISIBILITY_TIMEOUT:
                target_visible = True
        
        dist = self.dist_to_point(self.robot_pose, self.goal_pose)
        
        # Feedback intermedio
        if self.total_steps % self.METRICS_UPDATE_RATE == 0:
            status_msg = "TRACKING" if target_visible else "LOST"
            msg_str = String()
            msg_str.data = f"Step {self.total_steps}: {status_msg} - Dist {dist:.2f}m"
            self.feedback_pub.publish(msg_str)
        
        msg_dist = Float32()
        msg_dist.data = dist
        self.distance_pub.publish(msg_dist)

        best_u = self.velocity_command()
        
        cmd = Twist()
        cmd.linear.x = float(best_u[0])
        cmd.angular.z = float(best_u[1])
        self.cmd_vel_pub.publish(cmd)
        
        # Passiamo anche stato visibilità per metriche corrette
        self.update_metrics(dist, float(best_u[0]), float(best_u[1]), target_visible)

        dist = self.dist_to_point(self.robot_pose, self.goal_pose)
        if dist < self.OPTIMAL_DIST:
            self.get_logger().info("Goal Reached!")
            self.stop()
            return

    ####################################
    ##            METRICS             ##
    ####################################

    def update_metrics(self, current_dist, v, w, target_visible):
        """
        Computes and logs metrics.
        """
        # Bearing error
        target_angle = np.arctan2(self.goal_pose[1]-self.robot_pose[1], 
                                  self.goal_pose[0]-self.robot_pose[0])
        bearing_error = normalize_angle(target_angle - self.robot_pose[2])

        # FIX: Tracking logic corretta
        is_tracking = 0
        if target_visible and current_dist <= self.TRACKING_MAX_DIST:
            self.tracked_steps += 1
            is_tracking = 1

        # RMSE Calc
        dist_error = current_dist - self.OPTIMAL_DIST
        self.sq_error_dist += (dist_error ** 2)
        self.sq_error_bearing += (bearing_error ** 2)

        # Lidar Stats
        current_min = float('inf')
        valid_lasers = self.lasers[np.isfinite(self.lasers)]
        if len(valid_lasers) > 0:
            current_min = np.min(valid_lasers)
            current_avg = np.mean(valid_lasers)
            
            if current_min < self.global_min_obstacle_dist:
                self.global_min_obstacle_dist = current_min
            
            self.cumulative_avg_lidar_dist += current_avg

        # --- LOG DATA TO CSV ---
        timestamp = self.get_clock().now().nanoseconds / 1e9
        # is_tracking ora è definito
        self.csv_writer.writerow([timestamp, current_dist, bearing_error, current_min, v, w, is_tracking, int(target_visible)])

        # Stampa a video ogni tanto
        current_time = self.get_clock().now().nanoseconds / 1e9
        if (current_time - self.last_print_time) > 2.0:
            self.print_metrics_report()
            self.last_print_time = current_time

    def print_metrics_report(self):
        if self.total_steps == 0: return
        
        tracking_pct = (self.tracked_steps / self.total_steps) * 100.0
        rmse_dist = np.sqrt(self.sq_error_dist / self.total_steps)
        rmse_bearing = np.sqrt(self.sq_error_bearing / self.total_steps)
        
        print(f"Tracking: {tracking_pct:.1f}% | RMSE Dist: {rmse_dist:.3f}m | Min Obs: {self.global_min_obstacle_dist:.2f}m")

    def save_final_report(self):
        """
        Called on node shutdown to save the summary to a text file.
        """
        # Chiudiamo il file CSV
        if self.csv_file:
            self.csv_file.close()

        if self.total_steps == 0: return

        tracking_pct = (self.tracked_steps / self.total_steps) * 100.0
        rmse_dist = np.sqrt(self.sq_error_dist / self.total_steps)
        rmse_bearing = np.sqrt(self.sq_error_bearing / self.total_steps)
        avg_lidar = self.cumulative_avg_lidar_dist / self.total_steps
        
        report = (
            f"--- FINAL REPORT TASK 3 ---\n"
            f"Date: {datetime.now()}\n"
            f"Total Steps: {self.total_steps}\n"
            f"Tracking Success: {tracking_pct:.2f}%\n"
            f"RMSE Distance (from optimal {self.OPTIMAL_DIST}m): {rmse_dist:.4f} m\n"
            f"RMSE Bearing: {rmse_bearing:.4f} rad\n"
            f"Minimum Obstacle Distance Encountered: {self.global_min_obstacle_dist:.4f} m\n"
            f"Average Lidar Distance: {avg_lidar:.4f} m\n"
            f"Total Collisions: {self.collision_count}\n"
        )
        
        with open(self.txt_filename, "w") as f:
            f.write(report)
        
        self.get_logger().info(f"Report saved to {self.txt_filename}")
        self.get_logger().info(f"Data log saved to {self.csv_filename}")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save metrics on exit
        node.save_final_report()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
