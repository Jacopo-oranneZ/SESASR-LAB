import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import math
import sys
import os

try:
    from rosbag2_reader_py import Rosbag2Reader
except ImportError:
    print("ERRORE: Manca 'rosbag2_reader_py.py'.")
    sys.exit(1)

# --- CONFIGURAZIONE ---
# Inserisci qui il path della bag DI SIMULAZIONE
BAG_PATH_DEFAULT = '/home/filo/ros2_ws/SESASR-LAB/LAB0801/rosbag2_simulazione' 

# Parametri Simulazione
MAX_LIDAR_DIST = 3.5

# --- RENDERING "SMART EXPOSURE" ---
SCAN_DECIMATION = 2       # Alta densità
POINT_SIZE = 2            # Punti precisi
POINT_ALPHA = 0.02        # Bassissima opacità per evidenziare i muri statici

# Filtro "Anti-Cubo" (Rimuove il target mobile dalla mappa ostacoli)
TARGET_RADIUS_FILTER = 0.5  # 50cm attorno al centro del cubo rosso

def get_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def dist_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

def main():
    bag_file = sys.argv[1] if len(sys.argv) > 1 else BAG_PATH_DEFAULT
    if not os.path.exists(bag_file):
        print(f"ERRORE PATH: {bag_file}")
        sys.exit(1)

    print(f"--- GENERAZIONE DASHBOARD SIMULAZIONE: {os.path.basename(bag_file)} ---")
    reader = Rosbag2Reader(bag_file)

    # Liste Dati
    gt_t, gt_x, gt_y = [], [], []         # Robot (Ground Truth)
    target_x, target_y = [], []           # Target (Dynamic Goal)
    cmd_t, cmd_v, cmd_w = [], [], []      # Comandi
    obs_x, obs_y = [], []                 # Ostacoli (LiDAR)

    # Stato Corrente
    curr_x, curr_y, curr_yaw = 0.0, 0.0, 0.0
    curr_target_x, curr_target_y = None, None
    
    has_gt = False
    start_time = None
    scan_count = 0

    for topic, msg, t in reader:
        if start_time is None: start_time = t
        rel_time = (t - start_time) * 1e-9

        # A. GROUND TRUTH (Robot Pose)
        if topic == '/ground_truth':
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            curr_x = pos.x
            curr_y = pos.y
            curr_yaw = get_yaw(ori)
            
            gt_t.append(rel_time)
            gt_x.append(curr_x)
            gt_y.append(curr_y)
            has_gt = True

        # B. DYNAMIC GOAL POSE (Target Reale)
        elif topic == '/dynamic_goal_pose':
            t_pos = msg.pose.pose.position
            curr_target_x = t_pos.x
            curr_target_y = t_pos.y
            
            target_x.append(curr_target_x)
            target_y.append(curr_target_y)

        # C. COMANDI
        elif topic == '/cmd_vel':
            cmd_t.append(rel_time)
            cmd_v.append(msg.linear.x)
            cmd_w.append(msg.angular.z)

        # D. SCAN (Con Filtro su Dynamic Goal)
        elif topic == '/scan' and has_gt:
            scan_count += 1
            if scan_count % SCAN_DECIMATION != 0: continue
            
            angle = msg.angle_min
            angle_inc = msg.angle_increment
            
            # Attiva filtro se conosciamo la posizione del target
            filter_active = (curr_target_x is not None)
            target_radius_sq = TARGET_RADIUS_FILTER**2

            for r in msg.ranges:
                if 0.1 < r < MAX_LIDAR_DIST:
                    # Proiezione geometrica usando la Ground Truth del robot
                    abs_angle = curr_yaw + angle
                    ox = curr_x + r * math.cos(abs_angle)
                    oy = curr_y + r * math.sin(abs_angle)
                    
                    # FILTRO ANTI-CUBO:
                    # Se il punto LiDAR cade sul cubo rosso, non disegnarlo come muro
                    if filter_active:
                        if dist_sq(ox, oy, curr_target_x, curr_target_y) < target_radius_sq:
                            angle += angle_inc
                            continue
                    
                    obs_x.append(ox)
                    obs_y.append(oy)
                
                angle += angle_inc

    # --- PLOTTING ---
    print(f"Plotting Simulation Data ({len(obs_x)} scan points)...")
    
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1]) 

    # 1. MAPPA
    ax_map = plt.subplot(gs[:, 0])
    
    # Ostacoli Statici (Muri) - Nero trasparente
    ax_map.scatter(obs_x, obs_y, s=POINT_SIZE, c='black', alpha=POINT_ALPHA, zorder=1, linewidths=0)
    
    # Traiettoria Target (Linea Marrone Tratteggiata)
    ax_map.plot(target_x, target_y, color='#8B4513', linewidth=2, linestyle='--', label='Dynamic Goal Path', zorder=2)
    
    # Traiettoria Robot (Linea Blu Solida)
    ax_map.plot(gt_x, gt_y, color='blue', linewidth=2, label='Robot Ground Truth', zorder=3)
    
    # Start / End Markers
    if gt_x:
        ax_map.plot(gt_x[0], gt_y[0], 'go', markersize=8, zorder=4) # Start
        ax_map.plot(gt_x[-1], gt_y[-1], 'rx', markersize=12, markeredgewidth=3, zorder=5) # End

    ax_map.set_title("Simulation: Ground Truth, Dynamic Goal & Environment", fontweight='bold')
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.axis('equal')
    ax_map.grid(True, linestyle='--', alpha=0.5)

    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Static Walls', 
               markerfacecolor='black', markersize=5),
        Line2D([0], [0], color='blue', lw=2, label='Robot (Ground Truth)'),
        Line2D([0], [0], color='#8B4513', lw=2, linestyle='--', label='Target (Dynamic Goal)'),
        Line2D([0], [0], marker='o', color='w', label='Start', 
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='x', color='w', label='End', 
               markeredgecolor='red', markersize=10, markeredgewidth=3)
    ]
    ax_map.legend(handles=legend_elements, loc='upper right')

    # 2. VELOCITÀ LINEARE
    ax_v = plt.subplot(gs[0, 1])
    if cmd_t: ax_v.plot(cmd_t, cmd_v, 'k-', label='Cmd Linear v')
    ax_v.set_title("Linear Velocity", fontweight='bold')
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True, linestyle=':')
    ax_v.legend(loc='upper right')
    ax_v.axhline(y=0.22, color='r', linestyle='--', alpha=0.3, label='Max Speed')

    # 3. VELOCITÀ ANGOLARE
    ax_w = plt.subplot(gs[1, 1])
    if cmd_t: ax_w.plot(cmd_t, cmd_w, 'g-', label='Cmd Angular w')
    ax_w.set_title("Angular Velocity", fontweight='bold')
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("w [rad/s]")
    ax_w.grid(True, linestyle=':')
    ax_w.legend(loc='upper right')

    plt.tight_layout()
    filename = "sim_dashboard_final.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafico salvato: {filename}")
    plt.show()

if __name__ == "__main__":
    main()