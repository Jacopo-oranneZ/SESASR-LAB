import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from lab04_pkg.Task0 import eval_gux


# --- 1. MODELLO DI MOTO (Sampling usando eval_gux) ---

def sample_motion_model_velocity(x_t_prev, u, alpha, dt):
    """
    Campiona una possibile posa futura usando eval_gux di Task0.
    """
    v, w = u
    
    # 1. Generiamo il comando "rumoroso"
    # La varianza dipende dalla velocità stessa
    var_v = alpha[0] * v**2 + alpha[1] * w**2
    var_w = alpha[2] * v**2 + alpha[3] * w**2
    
    # Aggiungiamo un epsilon per stabilità numerica
    v_hat = v + np.random.normal(0, np.sqrt(var_v + 1e-10))
    w_hat = w + np.random.normal(0, np.sqrt(var_w + 1e-10))
    
    u_noisy = np.array([v_hat, w_hat])
    
    # 2. Usiamo la funzione cinematica del Task 0 per calcolare la posa
    # eval_gux accetta: (mu, u, sigma_u, dt)
    # sigma_u qui non serve a eval_gux (che calcola la media), quindi passiamo dummy
    x_next = eval_gux(x_t_prev, u_noisy, None, dt)
        
    return x_next

def plot_motion_samples():
    x0 = np.array([0.0, 0.0, 0.0]) # Partenza
    u = np.array([1.0, 1.0])       # Comando: v=1, w=1
    dt = 1.0
    n_samples = 500

    # Parametri Rumore (Alpha)
    alpha_angular = [0.01, 0.01, 0.1, 0.5]  # Alto rumore su w
    alpha_linear = [0.5, 0.1, 0.01, 0.01]   # Alto rumore su v

    plt.figure(figsize=(12, 5))

    # Plot A: Errore Angolare
    plt.subplot(1, 2, 1)
    samples_a = [sample_motion_model_velocity(x0, u, alpha_angular, dt) for _ in range(n_samples)]
    samples_a = np.array(samples_a)
    plt.plot(samples_a[:, 0], samples_a[:, 1], 'r.')
    plt.plot(x0[0], x0[1], 'ko', label="Start", markersize=8)

    plt.title(f"High angular uncertainty")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.grid(True)

    # Plot B: Errore Lineare
    plt.subplot(1, 2, 2)
    samples_b = [sample_motion_model_velocity(x0, u, alpha_linear, dt) for _ in range(n_samples)]
    samples_b = np.array(samples_b)
    plt.plot(samples_b[:, 0], samples_b[:, 1], 'b.')
    plt.plot(x0[0], x0[1], 'ko', label="Start", markersize=8)

    plt.title(f"High linear uncertainty")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.grid(True)

    plt.savefig("task0_motion_samples.png")
    plt.show()
    print("Salvato task0_motion_samples.png")

# --- 2. MODELLO DI MISURA (Inverse Sampling) ---

def sample_measurement_model_inverse(x_state, landmark, z, sigma):
    """
    Inverte il modello di misura h(x) per trovare dove potrebbe essere il robot.
    """
    # 1. Aggiungiamo rumore alla misura z
    r_noisy = z[0] + np.random.normal(0, sigma[0])
    phi_noisy = z[1] + np.random.normal(0, sigma[1])
    
    # 2. Inversione geometrica
    # Dato che h(x) = [dist, atan2(dy, dx) - theta]
    # Allora:
    # x = mx - r * cos(theta + phi)
    # y = my - r * sin(theta + phi)
    
    theta = x_state[2] # Assumiamo orientamento noto per semplificare il plot 2D
    
    x_est = landmark[0] - r_noisy * np.cos(theta + phi_noisy)
    y_est = landmark[1] - r_noisy * np.sin(theta + phi_noisy)
    
    return np.array([x_est, y_est])

def plot_measurement_samples():
    print("Generazione plot Modello Misura...")
    landmark = np.array([2.0, 2.0])
    true_pose = np.array([0.0, 0.0, 0.0])
    
    # Misura ideale
    dx = landmark[0] - true_pose[0]
    dy = landmark[1] - true_pose[1]
    z_true = [math.sqrt(dx**2 + dy**2), math.atan2(dy, dx) - true_pose[2]]
    
    sigma = [0.3, 0.15] # Rumore [0.3m, 0.15rad]
    n_samples = 1000

    samples = [sample_measurement_model_inverse(true_pose, landmark, z_true, sigma) for _ in range(n_samples)]
    samples = np.array(samples)

    plt.figure(figsize=(6, 6))
    plt.plot(landmark[0], landmark[1], 'k*', markersize=12, label="Landmark")
    plt.plot(samples[:, 0], samples[:, 1], 'c.', label="Possible estimates")
    plt.plot(true_pose[0], true_pose[1], 'go', markersize=10, label="True position")
    
    plt.title(f"Localization uncertainty , Landmark\nSigma={sigma}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    plt.savefig("task0_measurement_samples.png")
    plt.show()
    print("Salvato task0_measurement_samples.png")

if __name__ == "__main__":
    plot_motion_samples()
    plot_measurement_samples()