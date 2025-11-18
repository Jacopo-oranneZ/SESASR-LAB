import math
import numpy as np
import sympy
from sympy import Matrix, symbols, cos, sin

# Definiamo i simboli globalmente per poterli usare ovunque
mx, my, x, y, theta, v, w, dt = symbols('mx my x y theta v w dt')

# Parametri di rumore predefiniti (se servono altrove)
a = [0.1, 0.01, 0.01, 0.1] 

def get_motion_jacobians():
    """
    Genera i Jacobiani simbolici una volta sola all'importazione del modulo.
    """
    # Modello velocità (approssimazione per w != 0, il limite w->0 è gestito separatamente)
    # Nota: Usiamo la cinematica standard. 
    # x' = x - (v/w)sin(theta) + (v/w)sin(theta + w*dt)
    # y' = y + (v/w)cos(theta) - (v/w)cos(theta + w*dt)
    # theta' = theta + w*dt
    
    R = v / w
    beta = theta + w * dt
    
    # Funzione vettoriale g(u, x) simbolica
    gux_sym = Matrix([
        x - R * sympy.sin(theta) + R * sympy.sin(beta),
        y + R * sympy.cos(theta) - R * sympy.cos(beta),
        theta + w * dt
    ])

    # Jacobiani Gt (rispetto allo stato) e Vt (rispetto al comando)
    Gt_sym = gux_sym.jacobian(Matrix([x, y, theta]))
    Vt_sym = gux_sym.jacobian(Matrix([v, w]))

    # Funzione di misura h(x)
    # Range = sqrt((mx-x)^2 + (my-y)^2)
    # Bearing = atan2(my-y, mx-x) - theta
    hx_sym = Matrix([
        sympy.sqrt((mx - x) ** 2 + (my - y) ** 2),
        sympy.atan2(my - y, mx - x) - theta
    ])
    
    Ht_sym = hx_sym.jacobian(Matrix([x, y, theta]))

    # LAMBDIFY: Trasforma le espressioni simboliche in funzioni Python veloci (numpy)
    # NOTA: L'ordine degli argomenti deve essere coerente con come li chiami in EKF!
    
    # Gt e Vt chiamati con: (*mu, *u, dt) -> (x, y, theta, v, w, dt)
    _eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt_sym, "numpy")
    _eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt_sym, "numpy")
    
    # Ht chiamato con: (*mu, mx, my) -> (x, y, theta, mx, my)
    _eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht_sym, "numpy")

    return _eval_Gt, _eval_Vt, _eval_Ht

# ------------------------------------------------------------------------------
# 1. DEFINIZIONE DELLE FUNZIONI GLOBALI (EXPORT)
# ------------------------------------------------------------------------------

# Generiamo le funzioni Jacobiane ora, così sono pronte per l'import
eval_Gt, eval_Vt, eval_Ht = get_motion_jacobians()

# Definiamo manualmente eval_gux per gestire i vettori e il caso w=0
# La classe EKF chiama: eval_gux(mu, u, sigma_u, dt)
def eval_gux(mu, u, sigma_u, dt):
    """
    Modello di movimento deterministico (Noise-Free) per la predizione della media.
    Gestisce il caso di velocità angolare nulla.
    """
    x_val, y_val, theta_val = mu
    v_val, w_val = u
    
    # Gestione singolarità w ~ 0 (moto rettilineo)
    if abs(w_val) < 1e-6:
        # Limite per w -> 0:
        # x' = x + v * cos(theta) * dt
        # y' = y + v * sin(theta) * dt
        # theta' = theta
        x_new = x_val + v_val * np.cos(theta_val) * dt
        y_new = y_val + v_val * np.sin(theta_val) * dt
        theta_new = theta_val + w_val * dt # w è piccolissimo ma lo aggiungiamo
    else:
        # Moto circolare standard
        r = v_val / w_val
        x_new = x_val - r * np.sin(theta_val) + r * np.sin(theta_val + w_val * dt)
        y_new = y_val + r * np.cos(theta_val) - r * np.cos(theta_val + w_val * dt)
        theta_new = theta_val + w_val * dt

    return np.array([x_new, y_new, theta_new])

# ------------------------------------------------------------------------------
# 2. FUNZIONI UTILS TASK 0 (Sensori e Sampling)
# ------------------------------------------------------------------------------

def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    # Versione dummy per compatibilità import, se usata
    m_x, m_y = landmark[:]
    x_r, y_r, theta_r = robot_pose[:]
    
    dist = math.sqrt((m_x - x_r)**2 + (m_y - y_r)**2)
    angle = math.atan2(m_y - y_r, m_x - x_r) - theta_r
    
    # Normalizza angolo
    angle = math.atan2(math.sin(angle), math.cos(angle))
    
    return np.array([dist, angle])

def compute_p_hit_dist(z, max_range, sigma):
    # Se serve per il sampling del task 0
    pass