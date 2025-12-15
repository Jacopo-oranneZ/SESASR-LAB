import math
import numpy as np
import sympy
from sympy import Matrix, symbols, cos, sin

# Definiamo i simboli globalmente per SymPy
mx, my, x, y, theta, v, w, dt = symbols('mx my x y theta v w dt')

"""
    QUESTO FILE CONTIENE LE FUNZIONI MATEMATICHE PER L'EKF 5D
    (Modello di moto a velocità costante e modelli di misura per
    landmark, odometria e IMU).

"""


def get_symbolic_functions_task2():
    """
    
    Genera le funzioni Python dai modelli simbolici per il Task 2.
    Stato aumentato: [x, y, theta, v, w]

    """
    
    R = v / w
    beta = theta + w * dt
    
    # Funzione g(x) simbolica (5x1)
    g_sym = Matrix([
        x - R * sympy.sin(theta) + R * sympy.sin(beta), # x'
        y + R * sympy.cos(theta) - R * sympy.cos(beta), # y'
        theta + w * dt,                                 # theta'
        v,                                              # v'
        w                                               # w'
    ])
    
    # Jacobiano Gt (5x5) rispetto allo stato [x, y, theta, v, w]
    Gt_sym = g_sym.jacobian(Matrix([x, y, theta, v, w]))
    
    
    # LANDMARK (Range, Bearing)
    # Dipende solo dalla posa (x, y, theta)
    h_land_sym = Matrix([
        sympy.sqrt((mx - x) ** 2 + (my - y) ** 2),
        sympy.atan2(my - y, mx - x) - theta
    ])
    # Jacobiano Ht_land (2x5) - Le colonne per v e w saranno zero
    H_land_sym = h_land_sym.jacobian(Matrix([x, y, theta, v, w]))

    # Misura diretta delle velocità
    h_odom_sym = Matrix([v, w])
    H_odom_sym = h_odom_sym.jacobian(Matrix([x, y, theta, v, w]))

    # Misura diretta velocità angolare
    h_imu_sym = Matrix([w])
    H_imu_sym = h_imu_sym.jacobian(Matrix([x, y, theta, v, w]))

    
    # Gt accetta (x, y, theta, v, w, dt)
    _raw_Gt_5d = sympy.lambdify((x, y, theta, v, w, dt), Gt_sym, "numpy")
    
    # H_land accetta (x, y, theta, mx, my) - Nota: v, w non servono qui matematicamente
    _eval_H_land_5d = sympy.lambdify((x, y, theta, mx, my), H_land_sym, "numpy")
    
    # H_odom e H_imu sono matrici costanti (selezione), non dipendono da nulla
    _eval_H_odom_5d = sympy.lambdify((), H_odom_sym, "numpy") 
    _eval_H_imu_5d  = sympy.lambdify((), H_imu_sym, "numpy")

    return _raw_Gt_5d, _eval_H_land_5d, _eval_H_odom_5d, _eval_H_imu_5d

# Inizializziamo le funzioni "raw"  all'avvio del modulo. Esse contengono singolarità (1/w).
_raw_Gt_5d, _raw_H_land_5d, _raw_H_odom_5d, _raw_H_imu_5d = get_symbolic_functions_task2()



###############################################
##     FUNZIONI MATEMATICHE PER L'EKF 5D    ##
###############################################

def eval_gux_5d(mu, u, sigma_u, dt):
    """
    Calcola la predizione dello stato (Media).
    Gestisce la singolarità w=0.
    """
    # Scompattiamo lo stato 5D
    x_val, y_val, theta_val, v_val, w_val = mu
    
    if abs(w_val) < 1e-6:
        # Modello Rettilineo (Limite w->0)
        x_new = x_val + v_val * dt * np.cos(theta_val)
        y_new = y_val + v_val * dt * np.sin(theta_val)
        theta_new = theta_val + w_val * dt
    else:
        # Modello Curvilineo (Standard)
        r = v_val / w_val
        x_new = x_val - r * np.sin(theta_val) + r * np.sin(theta_val + w_val * dt)
        y_new = y_val + r * np.cos(theta_val) - r * np.cos(theta_val + w_val * dt)
        theta_new = theta_val + w_val * dt
        
    # v e w rimangono costanti (Random Walk model)
    return np.array([x_new, y_new, theta_new, v_val, w_val])



##############################################
##     FUNZIONI JACOBIANI MISURE EKF 5D     ##
##############################################

def eval_Gt_5d(x_val, y_val, theta_val, v_val, w_val, dt_val):
    """
    Jacobiano del Moto 5D.
    Argomenti: 5 stati + 1 dt = 6 argomenti.
    (Coerente con nodo che passa u=None)
    """
    if abs(w_val) < 1e-6:
        # Jacobiano per moto rettilineo (evita divisione per zero)
        ss, cc = np.sin(theta_val), np.cos(theta_val)
        
        # Matrice 5x5 identità di base
        Gt = np.eye(5)
        
        # Termini derivati da x' = x + v*dt*cos(theta) ...
        # d(x')/d(theta)
        Gt[0, 2] = -v_val * dt_val * ss
        # d(x')/d(v)
        Gt[0, 3] = dt_val * cc
        
        # d(y')/d(theta)
        Gt[1, 2] =  v_val * dt_val * cc
        # d(y')/d(v)
        Gt[1, 3] = dt_val * ss
        
        # d(theta')/d(w)
        Gt[2, 4] = dt_val
        
        return Gt
    else:
        # Usa la formula esatta di SymPy
        return _raw_Gt_5d(x_val, y_val, theta_val, v_val, w_val, dt_val)
    


def eval_H_land_5d(x_val, y_val, theta_val, v_val, w_val, mx_val, my_val):
    """

    Jacobiano Misura Landmark 5D.
    Accetta 7 argomenti (5 stato + 2 landmark),
    anche se 'v_val' e 'w_val' non vengono usati matematicamente.
    Questo serve perché l'EKF generico passa tutto il vettore di stato mu.

    """
    return _raw_H_land_5d(x_val, y_val, theta_val, mx_val, my_val)

def eval_H_odom_5d():
    """

    Jacobiano Misura Odometria (Costante).

    """
    return _raw_H_odom_5d()

def eval_H_imu_5d():
    """

    Jacobiano Misura IMU (Costante).

    """
    return _raw_H_imu_5d()