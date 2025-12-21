import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




def plot_localization_data(csv_file="localization_data.csv"):
    
    # 1. Lettura dei dati
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Errore: File non trovato. Assicurati che '{csv_file}' sia presente.")
        return
    
    # Separazione dei dati per sorgente
    odom_df = df[df['source'] == 'odom']
    ekf_df = df[df['source'] == 'ekf']
    gt_df = df[df['source'] == 'gt']  


    # --- PLOT 1: Stato (x, y, yaw) in funzione del Tempo ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('Odom, Ground Truth, EKF Comparison', fontsize=16)

    # Componente X - CORREZIONE: Usiamo .to_numpy()
    axes[2].plot(odom_df['time'].to_numpy(), odom_df['x'].to_numpy() - 2.0, label='X', linestyle='--', color='blue')
    axes[2].plot(odom_df['time'].to_numpy(), odom_df['y'].to_numpy() - 0.5, label='Y', color='red')
    axes[2].plot(odom_df['time'].to_numpy(), odom_df['yaw'].to_numpy(), label=' Yaw', color='green')
    axes[2].set_ylabel('Odom')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True)
    axes[2].legend()

    # Componente Y - CORREZIONE: Usiamo .to_numpy()
    axes[1].plot(gt_df['time'].to_numpy(), gt_df['x'].to_numpy(), label=' X', linestyle='--', color='blue')
    axes[1].plot(gt_df['time'].to_numpy(), gt_df['y'].to_numpy(), label=' Y', color='red')
    axes[1].plot(gt_df['time'].to_numpy(), gt_df['yaw'].to_numpy(), label=' Yaw', color='green')
    axes[1].set_ylabel('Ground Truth')
    axes[1].grid(True)
    axes[1].legend()

    # Componente Yaw (Orientazione) - CORREZIONE: Usiamo .to_numpy()
    axes[0].plot(ekf_df['time'].to_numpy(), ekf_df['x'].to_numpy(), label=' X ', linestyle='--', color='blue')
    axes[0].plot(ekf_df['time'].to_numpy(), ekf_df['y'].to_numpy(), label=' Y ', color='red')
    axes[0].plot(ekf_df['time'].to_numpy(), ekf_df['yaw'].to_numpy(), label=' Yaw', color='green')
    axes[0].set_ylabel('EKF')
    axes[0].grid(True)
    axes[0].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    plt.show()

    # --- PLOT 2: Traiettoria (X, Y) ---
    plt.figure(figsize=(10, 10))
    # CORREZIONE: Usiamo .to_numpy() anche qui
    plt.plot(odom_df['x'].to_numpy(), odom_df['y'].to_numpy(), label=' Odom Trajectory', linestyle='--', color='blue')
    plt.plot(ekf_df['x'].to_numpy(), ekf_df['y'].to_numpy(), label=' EKF Trajectory', color='red')
    plt.plot(gt_df['x'].to_numpy(), gt_df['y'].to_numpy(), label=' GT Trajectory', color='green')

    # Aggiungi punti iniziali/finali per chiarezza
    # Per i singoli punti, l'indexing di Pandas va bene, ma per coerenza possiamo usare .iloc[i].item()
    plt.plot(odom_df['x'].iloc[0], odom_df['y'].iloc[0], 'o', markersize=8, color='lightblue', label='Start Odom')
    plt.plot(ekf_df['x'].iloc[0], ekf_df['y'].iloc[0], 'o', markersize=8, color='lightcoral', label='Start EKF')
    plt.plot(gt_df['x'].iloc[0], gt_df['y'].iloc[0], 'o', markersize=8, color='lightgreen', label='Start GT')
    plt.plot(odom_df['x'].iloc[-1], odom_df['y'].iloc[-1], 'X', markersize=8, color='darkblue', label='End Odom')
    plt.plot(ekf_df['x'].iloc[-1], ekf_df['y'].iloc[-1], 'X', markersize=8, color='darkred', label='End EKF')
    plt.plot(gt_df['x'].iloc[-1], gt_df['y'].iloc[-1], 'X', markersize=8, color='darkgreen', label='End GT')

    
    

    
    plt.xlabel('Position X [m]')
    plt.ylabel('Position Y [m]')
    plt.title('Spatial Trajectory Comparison')
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.grid(True)
    plt.legend()
    plt.show()


    plt.figure(figsize=(10, 10))
    plt.plot(odom_df['time'].to_numpy(), odom_df['yaw'].to_numpy(), label=' Odom Trajectory', linestyle='--', color='blue')
    plt.plot(ekf_df['time'].to_numpy(), ekf_df['yaw'].to_numpy(), label=' EKF Trajectory', color='red')
    plt.plot(gt_df['time'].to_numpy(), gt_df['yaw'].to_numpy(), label=' GT Trajectory', color='green')

    plt.xlabel('Time [s]')
    plt.ylabel('Orientation Yaw [rad]')
    plt.title('Orientation Yaw Comparison')
    plt.grid(True)
    plt.legend()
    plt.show()
if __name__ == '__main__':
    plot_localization_data()