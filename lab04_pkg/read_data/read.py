from rosbag2_reader_py import Rosbag2Reader
import matplotlib.pyplot as plt
import numpy as np
import os
import math

"""
    QUESTO SCRIPT LEGGE I DATI DELLA ROSBAG SPECIFICATA,
    CALCOLA GLI ERRORI TRA LA GROUND TRUTH E LA STIMA DALL'EKF,
    E GENERA UN PLOT DELLA TRAIETTORIA CON I RISULTATI.

"""

BAG_PATH = '/home/l0dz/SESASR-LAB/src/lab04_pkg/read_data/rosbag_task2'  # MODIFICARE QUI SE NECESSARIO

def calculate_metrics(gt_data, est_data):
    """

    Calcola RMSE e MAE.
    gt_data: lista di tuple (timestamp, x, y)
    est_data: lista di tuple (timestamp, x, y)
    Restituisce: RMSE, MAE, gt_x_aligned, gt_y_aligned

    """
    # Convertiamo in numpy array per facilità
    gt = np.array(gt_data)   # [t, x, y]
    est = np.array(est_data) # [t, x, y]

    # Problema: GT ed EKF hanno numero di campioni diversi e tempi diversi.
    # Soluzione: Interpoliamo la Ground Truth sui timestamp della Stima.
    
    # Interpolazione lineare
    gt_x_interp = np.interp(est[:, 0], gt[:, 0], gt[:, 1])
    gt_y_interp = np.interp(est[:, 0], gt[:, 0], gt[:, 2])

    # Calcolo errore euclideo punto per punto
    dx = est[:, 1] - gt_x_interp
    dy = est[:, 2] - gt_y_interp
    error_dist = np.sqrt(dx**2 + dy**2)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(error_dist**2))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(error_dist)

    return rmse, mae, gt_x_interp, gt_y_interp

def main():
    """
    
    Funzione principale per leggere la rosbag, calcolare errori e generare il plot.

    """


    # Verifica esistenza
    if not os.path.exists(BAG_PATH):
        print(f"[ERRORE] La cartella non esiste: {BAG_PATH}")
        print("Suggerimento: Usa il percorso assoluto (es: /home/l0dz/...)")
        return

    print(f"Caricamento bag da: {BAG_PATH} ...")
    bag = Rosbag2Reader(BAG_PATH)

    # Liste per salvare i dati: (timestamp, x, y)
    gt_data = []
    ekf_data = []
    odom_data = []

    # --- LETTURA DATI ---
    for topic, msg, t in bag:
        # Timestamp in secondi (t è in nanosecondi)
        time_sec = t * 1e-9

        if topic == '/ground_truth':
            gt_data.append([time_sec, msg.pose.pose.position.x, msg.pose.pose.position.y])
            
        elif topic == '/ekf':
            ekf_data.append([time_sec, msg.pose.pose.position.x, msg.pose.pose.position.y])

        elif topic == '/odom':
            # Se vuoi confrontare anche l'odometria pura
            odom_data.append([time_sec, msg.pose.pose.position.x, msg.pose.pose.position.y])

    

    print(f"Dati estratti: {len(gt_data)} GT points, {len(ekf_data)} EKF points.")

    if len(gt_data) == 0 or len(ekf_data) == 0:
        print("Attenzione: Nessun dato trovato per GT o EKF. Controlla i nomi dei topic!")
        return

    # --- CALCOLO ERRORI ---
    rmse, mae, gt_x_aligned, gt_y_aligned = calculate_metrics(gt_data, ekf_data)

    print("-" * 30)
    print(f"RISULTATI ANALISI:")
    print(f"RMSE: {rmse:.4f} m")
    print(f"MAE:  {mae:.4f} m")
    print("-" * 30)

    # --- PLOTTING ---
    est_np = np.array(ekf_data)
    gt_np = np.array(gt_data)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Traiettoria XY
    plt.plot(gt_np[:, 1], gt_np[:, 2], 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.plot(est_np[:, 1], est_np[:, 2], 'b--', label='EKF Estimate', linewidth=1.5)
    
    # Plot Odom
    if len(odom_data) > 0:
        odom_np = np.array(odom_data)
        # L'odometria parte da 0,0 mentre il robot è altrove. 
        plt.plot(odom_np[:, 1], odom_np[:, 2], 'g:', label='Odometry (Raw)', alpha=0.5)

    plt.title(f'Robot Trajectory (RMSE: {rmse:.3f}m)')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal') # Fondamentale per vedere la geometria corretta
    plt.legend()
    plt.grid(True)
    
    # Salviamo il plot
    output_file = 'trajectory_analysis.png'
    plt.savefig(output_file)
    print(f"Grafico salvato come {output_file}")
    plt.show()

if __name__ == "__main__":
    main()