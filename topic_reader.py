import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from rosbag2_reader_py import Rosbag2Reader
import sys
import os

# --- Funzioni di supporto ---
def get_yaw_from_msg(msg):
    if hasattr(msg, 'pose') and hasattr(msg.pose, 'pose'):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    return 0.0

def get_time_sec(msg):
    if hasattr(msg, 'header'):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    return 0.0

def plot_dashboard(bag_path):
    print(f"Ingegnere: Generazione Dashboard per: {os.path.basename(bag_path)}")
    
    target_topics = ['/odom', '/ekf', '/ground_truth']
    
    try:
        reader = Rosbag2Reader(bag_path, topics_filter=target_topics)
    except Exception as e:
        print(f"Errore critico: {e}")
        return

    # Configurazioni Estetiche (Colori e Stili)
    # Ground Truth: Solido, Verde (Riferimento)
    # EKF: Solido, Blu (Stima filtrata)
    # Odom: Tratteggiato, Rosso (Stima grezza/drift)
    styles = {
        '/ground_truth': {'color': '#2ca02c', 'label': 'Ground Truth', 'ls': '-', 'lw': 2, 'alpha': 0.8},
        '/ekf':          {'color': '#1f77b4', 'label': 'EKF',          'ls': '-', 'lw': 1.5, 'alpha': 1.0},
        '/odom':         {'color': '#d62728', 'label': 'Odom',         'ls': '--','lw': 1.5, 'alpha': 0.7}
    }

    # Contenitori Dati
    data = {t: {'t': [], 'x': [], 'y': [], 'yaw': []} for t in target_topics}
    start_time = None

    print("Ingegnere: Estrazione dati...")
    count = 0
    for topic, msg, t in reader:
        if topic in data:
            current_time = get_time_sec(msg)
            if start_time is None: start_time = current_time
            
            data[topic]['t'].append(current_time - start_time)
            data[topic]['x'].append(msg.pose.pose.position.x)
            data[topic]['y'].append(msg.pose.pose.position.y)
            data[topic]['yaw'].append(get_yaw_from_msg(msg))
            count += 1

    print(f"Dati estratti: {count} messaggi.")

    # --- COSTRUZIONE DELLA DASHBOARD ---
    fig = plt.figure(figsize=(16, 9))
    # Layout: 3 righe, 2 colonne. 
    # La colonna 0 (Sinistra) è occupata interamente dal grafico spaziale.
    # La colonna 1 (Destra) è divisa in 3 righe per x, y, yaw.
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.5, 1]) 

    # Assi
    ax_spatial = fig.add_subplot(gs[:, 0]) # Tutta la colonna sinistra
    ax_x = fig.add_subplot(gs[0, 1])       # Alto destra
    ax_y = fig.add_subplot(gs[1, 1])       # Centro destra
    ax_yaw = fig.add_subplot(gs[2, 1])     # Basso destra

    # Titolo della Dashboard
    plt.suptitle(f"Localization Analysis: {os.path.basename(bag_path)}", fontsize=16, fontweight='bold')

    # Loop di plotting per ogni topic
    for topic in target_topics:
        if not data[topic]['t']: continue
        
        s = styles[topic] # Stile corrente
        d = data[topic]   # Dati correnti

        # 1. Plot Spaziale
        ax_spatial.plot(d['x'], d['y'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'], alpha=s['alpha'])
        
        # 2. Plot X
        ax_x.plot(d['t'], d['x'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'])
        
        # 3. Plot Y
        ax_y.plot(d['t'], d['y'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'])
        
        # 4. Plot Yaw
        ax_yaw.plot(d['t'], d['yaw'], label=s['label'], color=s['color'], ls=s['ls'], lw=s['lw'])

    # --- ESTETICA RAFFINATA ---

    # Grafico Spaziale
    ax_spatial.set_title("Trajectory Path (XY Plane)", fontweight='bold')
    ax_spatial.set_xlabel("x [m]")
    ax_spatial.set_ylabel("y [m]")
    ax_spatial.axis('equal') # Cruciale per la geometria
    ax_spatial.grid(True, linestyle=':', alpha=0.6)
    ax_spatial.legend(loc='best')
    # Aggiungi marker di start
    if data['/ground_truth']['x']:
        ax_spatial.plot(data['/ground_truth']['x'][0], data['/ground_truth']['y'][0], 'ko', label='Start')

    # Grafici Temporali (condividono l'asse X del tempo se si vuole, qui li lascio indipendenti per chiarezza)
    ax_x.set_title("Position x", fontweight='bold', pad=2)
    ax_x.set_ylabel("x [m]")
    ax_x.grid(True, linestyle=':', alpha=0.6)
    # Rimuoviamo le etichette dell'asse X per i grafici superiori per pulizia
    plt.setp(ax_x.get_xticklabels(), visible=False)

    ax_y.set_title("Position y", fontweight='bold', pad=2)
    ax_y.set_ylabel("y [m]")
    ax_y.grid(True, linestyle=':', alpha=0.6)
    plt.setp(ax_y.get_xticklabels(), visible=False)

    ax_yaw.set_title("Orientation \u03B8 (yaw)", fontweight='bold', pad=2)
    ax_yaw.set_ylabel("\u03B8 [rad]")
    ax_yaw.set_xlabel("time [s]")
    ax_yaw.grid(True, linestyle=':', alpha=0.6)
    
    # Una sola legenda per la colonna di destra (per non affollare)
    # ax_x.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Spazio per il titolo principale
    
    print("Ingegnere: Dashboard generata. Analizziamo i risultati!")
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