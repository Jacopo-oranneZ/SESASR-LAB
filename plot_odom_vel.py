import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# Inserisci qui il percorso della tua bag
BAG_PATH_DEFAULT = '/home/filo/ros2_ws/SESASR-LAB/LAB0801/rosbag2_2026_01_08-18_51_57'

# Parametri estratti dal tuo codice v3_avoid_obstacles.py
CAMERA_OFFSET_X = 0.05  # 5 cm offset in avanti

def get_yaw(q):
    """Converte quaternioni in angolo yaw (radianti)"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def main():
    bag_file = sys.argv[1] if len(sys.argv) > 1 else BAG_PATH_DEFAULT
    if not os.path.exists(bag_file):
        print(f"ERRORE PATH: {bag_file}")
        sys.exit(1)

    print(f"--- GENERAZIONE DASHBOARD REPORT: {os.path.basename(bag_file)} ---")
    reader = Rosbag2Reader(bag_file)

    # DATI PER I GRAFICI
    # 1. Traiettoria (da Odom)
    odom_t = []
    odom_x = []
    odom_y = []
    
    # 2. Velocità (da Cmd_vel) - NUOVE LISTE TEMPORALI DEDICATE
    cmd_t = [] 
    cmd_v = [] # Lineare
    cmd_w = [] # Angolare

    # 3. Landmark (Calcolati)
    lm_x = []
    lm_y = []

    # Variabili di stato per la trasformazione
    curr_x = 0.0
    curr_y = 0.0
    curr_yaw = 0.0
    has_odom = False
    start_time = None

    for topic, msg, t in reader:
        if start_time is None: start_time = t
        # Calcoliamo il tempo relativo per QUESTO messaggio specifico
        rel_time = (t - start_time) * 1e-9

        # --- A. ODOMETRIA (/odom) ---
        if topic == '/odom':
            # Posizione
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            curr_x = pos.x
            curr_y = pos.y
            curr_yaw = get_yaw(ori)

            # Salvataggio dati
            odom_t.append(rel_time)
            odom_x.append(curr_x)
            odom_y.append(curr_y)
            has_odom = True

        # --- B. COMANDI (/cmd_vel) ---
        elif topic == '/cmd_vel':
            # Velocità (Twist)
            v_lin = msg.linear.x
            v_ang = msg.angular.z

            # IMPORTANTE: Usiamo rel_time corrente, non quello dell'odom!
            cmd_t.append(rel_time) 
            cmd_v.append(v_lin)
            cmd_w.append(v_ang)

        # --- C. CAMERA (/camera/landmarks) ---
        elif topic == '/camera/landmarks' and has_odom:
            if hasattr(msg, 'landmarks'):
                for lm in msg.landmarks:
                    try:
                        # Estrazione dati grezzi
                        r = lm.range
                        b = lm.bearing

                        # Filtro base
                        if r < 0.01 or r > 5.0: continue

                        # --- TRASFORMAZIONE GEOMETRICA ---
                        # 1. Polare -> Cartesiana (Frame Robot)
                        rel_x = r * math.cos(b)
                        rel_y = r * math.sin(b)

                        # 2. Offset Camera
                        rel_x += CAMERA_OFFSET_X

                        # 3. Trasformazione Globale
                        glob_x = curr_x + (rel_x * math.cos(curr_yaw) - rel_y * math.sin(curr_yaw))
                        glob_y = curr_y + (rel_x * math.sin(curr_yaw) + rel_y * math.cos(curr_yaw))
                        
                        lm_x.append(glob_x)
                        lm_y.append(glob_y)

                    except AttributeError:
                        pass

    # --- D. PLOTTING ---
    print(f"Dati estratti -> Odom: {len(odom_x)}, Cmd: {len(cmd_v)}, Landmark: {len(lm_x)}")
    
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 1]) 

    # 1. MAPPA (Sinistra)
    ax_map = plt.subplot(gs[:, 0])
    ax_map.plot(odom_x, odom_y, label='Robot Trajectory', color='blue', linewidth=2, zorder=1)
    
    if odom_x:
        ax_map.plot(odom_x[0], odom_y[0], 'go', label='Start', zorder=3)
        ax_map.plot(odom_x[-1], odom_y[-1], 'rx', label='End', zorder=3)
    
    ax_map.scatter(lm_x, lm_y, c='orange', s=10, alpha=0.5, label='Detected Target', zorder=2)
    
    ax_map.set_title("Robot Path & Target Reconstruction", fontweight='bold')
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.axis('equal')
    ax_map.grid(True, linestyle='--')
    ax_map.legend(loc='upper right')

    # 2. VELOCITÀ LINEARE (Destra Alto)
    ax_v = plt.subplot(gs[0, 1])
    # QUI LA CORREZIONE: Usiamo cmd_t, non odom_t!
    if cmd_t:
        ax_v.plot(cmd_t, cmd_v, 'k-', label='Cmd Linear Velocity')
    else:
        print("ATTENZIONE: Nessun comando velocità trovato!")

    ax_v.set_title("Linear Velocity Profile", fontweight='bold')
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True, linestyle=':')
    ax_v.legend(loc='upper right')
    ax_v.axhline(y=0.22, color='r', linestyle='--', alpha=0.3, label='Max Speed')

    # 3. VELOCITÀ ANGOLARE (Destra Basso)
    ax_w = plt.subplot(gs[1, 1])
    # QUI LA CORREZIONE: Usiamo cmd_t, non odom_t!
    if cmd_t:
        ax_w.plot(cmd_t, cmd_w, 'g-', label='Cmd Angular Velocity')

    ax_w.set_title("Angular Velocity Profile", fontweight='bold')
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylabel("w [rad/s]")
    ax_w.grid(True, linestyle=':')
    ax_w.legend(loc='upper right')

    plt.tight_layout()
    filename = "final_dashboard_task3_v2.png"
    plt.savefig(filename, dpi=300)
    print(f"Grafico salvato con successo: {filename}")
    plt.show()

if __name__ == "__main__":
    main()