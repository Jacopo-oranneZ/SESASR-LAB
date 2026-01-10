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
BAG_PATH_DEFAULT = '/home/filo/ros2_ws/SESASR-LAB/LAB0801/rosbag2_2026_01_08-18_51_57'

# Parametri Robot
CAMERA_OFFSET_X = 0.05
MAX_LIDAR_DIST = 3.5

# --- PARAMETRI "LONG EXPOSURE" (Il segreto per la pulizia) ---
# Leggiamo molti dati, ma li rendiamo trasparentissimi.
# I muri si sovrappongono diventando neri. Il rumore resta invisibile.
SCAN_DECIMATION = 2      # Alta densità
POINT_SIZE = 2           # Punti precisi
POINT_ALPHA = 0.02       # 2% di opacità: serve sovrapporre 50 punti per fare un nero pieno!

# Filtro Anti-Umano (Più aggressivo)
HUMAN_RADIUS_FILTER = 0.75  # 75cm di raggio "no-scan" attorno al target
TARGET_MEMORY_TIME = 2.0    # Ricorda la posizione dell'umano più a lungo

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

    print(f"--- GENERAZIONE DASHBOARD V7 (SMART EXPOSURE): {os.path.basename(bag_file)} ---")
    print("Elaborazione ad alta densità in corso...")
    reader = Rosbag2Reader(bag_file)

    odom_t, odom_x, odom_y = [], [], []
    cmd_t, cmd_v, cmd_w = [], [], []
    lm_x, lm_y = [], []
    obs_x, obs_y = [], []
    
    curr_x, curr_y, curr_yaw = 0.0, 0.0, 0.0
    has_odom = False
    start_time = None
    scan_count = 0

    last_target_x = None
    last_target_y = None
    last_target_seen_time = -10.0

    for topic, msg, t in reader:
        if start_time is None: start_time = t
        rel_time = (t - start_time) * 1e-9

        # A. ODOMETRIA
        if topic == '/odom':
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            curr_x = pos.x
            curr_y = pos.y
            curr_yaw = get_yaw(ori)
            odom_t.append(rel_time)
            odom_x.append(curr_x)
            odom_y.append(curr_y)
            has_odom = True

        # B. COMANDI
        elif topic == '/cmd_vel':
            cmd_t.append(rel_time) 
            cmd_v.append(msg.linear.x)
            cmd_w.append(msg.angular.z)

        # C. CAMERA (Aggiorna posizione filtro umano)
        elif topic == '/camera/landmarks' and has_odom:
            if hasattr(msg, 'landmarks'):
                for lm in msg.landmarks:
                    try:
                        r = lm.range
                        b = lm.bearing
                        if r < 0.01 or r > 5.0: continue
                        
                        rel_x = r * math.cos(b) + CAMERA_OFFSET_X
                        rel_y = r * math.sin(b)
                        glob_x = curr_x + (rel_x * math.cos(curr_yaw) - rel_y * math.sin(curr_yaw))
                        glob_y = curr_y + (rel_x * math.sin(curr_yaw) + rel_y * math.cos(curr_yaw))
                        
                        lm_x.append(glob_x)
                        lm_y.append(glob_y)

                        last_target_x = glob_x
                        last_target_y = glob_y
                        last_target_seen_time = rel_time
                    except AttributeError: pass

        # D. SCAN (SMART RENDERING)
        elif topic == '/scan' and has_odom:
            scan_count += 1
            if scan_count % SCAN_DECIMATION != 0: continue
            
            angle = msg.angle_min
            angle_inc = msg.angle_increment
            
            # Attiva filtro se abbiamo visto il target di recente
            filter_active = False
            if last_target_x is not None and (rel_time - last_target_seen_time) < TARGET_MEMORY_TIME:
                filter_active = True
                human_radius_sq = HUMAN_RADIUS_FILTER**2

            for r in msg.ranges:
                if 0.1 < r < MAX_LIDAR_DIST:
                    abs_angle = curr_yaw + angle
                    ox = curr_x + r * math.cos(abs_angle)
                    oy = curr_y + r * math.sin(abs_angle)
                    
                    # Filtra Umano
                    if filter_active:
                        if dist_sq(ox, oy, last_target_x, last_target_y) < human_radius_sq:
                            angle += angle_inc
                            continue
                    
                    obs_x.append(ox)
                    obs_y.append(oy)
                
                angle += angle_inc

    # --- PLOTTING ---
    print(f"Plotting {len(obs_x)} scan points (questo richiederà qualche secondo)...")
    
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5, 1]) 

    # 1. MAPPA
    ax_map = plt.subplot(gs[:, 0])
    
    # TRUCCO PRO: Colore Nero Pieno (#000000) ma Alpha bassissimo (0.02)
    # Risultato: I muri diventano neri solidi. Il rumore resta grigio fumo invisibile.
    ax_map.scatter(obs_x, obs_y, s=POINT_SIZE, c='black', alpha=POINT_ALPHA, zorder=1, linewidths=0)
    
    ax_map.plot(odom_x, odom_y, color='blue', linewidth=2, zorder=2)
    if odom_x:
        ax_map.plot(odom_x[0], odom_y[0], 'go', zorder=4)
        ax_map.plot(odom_x[-1], odom_y[-1], 'rx', zorder=4)
    # Landmark un po' più vivaci
    ax_map.scatter(lm_x, lm_y, c='#FF8C00', s=20, alpha=0.9, edgecolors='k', linewidths=0.5, zorder=3)
    
    ax_map.set_title("Environment Reconstruction: Path, Target & Cleaned Obstacles (Smart)", fontweight='bold')
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.axis('equal')
    ax_map.grid(True, linestyle='--', alpha=0.5)
    
    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Static Obstacles', 
               markerfacecolor='black', markersize=5),
        Line2D([0], [0], color='blue', lw=2, label='Robot Trajectory'),
        Line2D([0], [0], marker='o', color='w', label='Start', 
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='End', 
               markerfacecolor='red', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Detected Target', 
               markerfacecolor='#FF8C00', markersize=6)
    ]
    ax_map.legend(handles=legend_elements, loc='upper right')

    # 2. VELOCITÀ
    ax_v = plt.subplot(gs[0, 1])
    if cmd_t: ax_v.plot(cmd_t, cmd_v, 'k-', label='Cmd Linear v')
    ax_v.set_title("Linear Velocity", fontweight='bold')
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True, linestyle=':')
    ax_v.legend(loc='upper right')
    ax_v.axhline(y=0.22, color='r', linestyle='--', alpha=0.3, label='Max Speed')

    ax_w = plt.subplot(gs[1, 1])
    if cmd_t: ax_w.plot(cmd_t, cmd_w, 'g-', label='Cmd Angular w')
    ax_w.set_title("Angular Velocity", fontweight='bold')
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("w [rad/s]")
    ax_w.grid(True, linestyle=':')
    ax_w.legend(loc='upper right')

    plt.tight_layout()
    filename = "final_dashboard_task3_v7_smart.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafico salvato: {filename}")
    plt.show()

if __name__ == "__main__":
    main()