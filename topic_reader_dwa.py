import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from rosbag2_reader_py import Rosbag2Reader
import sys
import os

# --- Funzioni di supporto ---
'''def get_yaw_from_msg(msg):
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    return 0.0
'''
def get_time_sec(msg):
    if hasattr(msg, 'header'):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return 0.0

def plot_dashboard(bag_path):
    print(f"Generazione Dashboard per: {os.path.basename(bag_path)}")
    
    target_topics = [#'/odom', '/ekf']
                      '/ground_truth']
    
    try:
        reader = Rosbag2Reader(bag_path, topics_filter=target_topics)
    except Exception as e:
        print(f"Errore critico: {e}")
        return

    styles = {
        #'/ekf':          {'color': "#d41313", 'label': 'EKF',          'ls': '-', 'lw': 2, 'alpha': 1.0},
        '/ground_truth': {'color': "#2ca02c6e", 'label': 'Ground Truth', 'ls': '-', 'lw': 2, 'alpha': 1.0},
        #'/odom':         {'color': "#100de0", 'label': 'Odom',         'ls': '--','lw': 2, 'alpha': 1.0}
    }

    # Contenitori Dati
    data = {t: {'t': [], 'x': [], 'y': [], 'yaw': [], 'v': [], 'w': []} for t in target_topics}
    start_time = None


    count = 0
    
    x_OFFSET = 0.0 #2.0
    y_OFFSET = 0.0 #0.5
   
    for topic, msg, t in reader:
        if topic in data:
            current_time = get_time_sec(msg)
            if start_time is None: start_time = current_time
            
            # Leggiamo i valori grezzi dal messaggio
            pos_x = msg.pose.pose.position.x
            pos_y = msg.pose.pose.position.y
            vel_v = msg.twist.twist.linear.x
            vel_w = msg.twist.twist.angular.z
            
            ### NUOVO: Logica di correzione
            if topic == '/ekf':
                pos_x -= x_OFFSET
                pos_y -= y_OFFSET
            #if topic == '/odom':
             #   pos_x -= x_OFFSET  # Aggiunge l'offset solo all'odometria
             #  pos_y -= y_OFFSET
            
            # Salviamo i valori (corretti o originali)
            data[topic]['t'].append(current_time - start_time)
            data[topic]['x'].append(pos_x) # <--- Ora usa la variabile pos_x (eventualmente corretta)
            data[topic]['y'].append(pos_y) # <--- Ora usa la variabile pos_y
            data[topic]['v'].append(vel_v)
            data[topic]['w'].append(vel_w)
            count += 1

    print(f"Dati estratti: {count} messaggi.")

    # --- COSTRUZIONE DELLA DASHBOARD ---
    fig = plt.figure(figsize=(13, 5))
    
    # Layout: 3 righe, 2 colonne. 
    # Colonna 0: Traiettoria Spaziale (XY)
    # Colonna 1: Velocità Lineare, Velocità Angolare, (Opzionale: Yaw o altro)
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.8, 1]) 

    # Assi
    ax_spatial = fig.add_subplot(gs[:, 0]) # Tutta la colonna sinistra
    ax_v = fig.add_subplot(gs[0, 1])       # Alto destra (Velocità Lineare)
    ax_w = fig.add_subplot(gs[1, 1])       # Centro destra (Velocità Angolare)
    # ax_yaw = fig.add_subplot(gs[2, 1])   # Basso destra (se vuoi riattivare lo Yaw)

    # Loop di plotting per ogni topic
    for topic in target_topics:
        if not data[topic]['t']: continue
        
        s = styles[topic] # Stile corrente
        d = data[topic]   # Dati correnti

        # 1. Grafico Spaziale (XY)
        ax_spatial.plot(d['x'], d['y'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'], alpha=s['alpha'])
        
        # Marker Start/End (Come hai fatto tu, ottimo tocco!)
        ax_spatial.plot(d['x'][0], d['y'][0], marker='o', color=s['color'], markersize=8, markeredgewidth=2.5, label='_nolegend_')
        ax_spatial.plot(d['x'][-1], d['y'][-1], marker='x', color=s['color'], markersize=8, markeredgewidth=2.5, label='_nolegend_')

        # 2. Plot Velocità Lineare (v)
        ax_v.plot(d['t'], d['v'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'])
        
        # 3. Plot Velocità Angolare (w)
        ax_w.plot(d['t'], d['w'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'])

    # --- ESTETICA ---
    ax_spatial.set_title("Spatial Trajectory", fontweight='bold')
    ax_spatial.set_xlabel("x [m]")
    ax_spatial.set_ylabel("y [m]")
    ax_spatial.axis('equal')
    ax_spatial.grid(True, linestyle=':', alpha=0.6)
    ax_spatial.legend(loc='best')

    # Estetica Velocità
    ax_v.set_title("Linear Velocity", fontweight='bold', pad=2)
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True, linestyle=':', alpha=0.6)
    # Nascondiamo le x labels se condividono l'asse, altrimenti lascia stare
    
    ax_w.set_title("Angular Velocity", fontweight='bold', pad=2)
    ax_w.set_ylabel("w [rad/s]")
    ax_w.set_xlabel("time [s]") # Solo l'ultimo grafico ha l'etichetta temporale
    ax_w.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    print("Ingegnere: Dashboard aggiornata con le velocità!")
    plt.show()
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]
    else:
        bag_file = input("Inserisci percorso bag: ")
    
    if os.path.exists(bag_file):
        plot_dashboard(bag_file)
    else:
        print("Path errato.")