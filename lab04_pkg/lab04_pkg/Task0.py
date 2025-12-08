import math
import numpy as np
import sympy
from sympy import Matrix, symbols, cos, sin

# Definiamo i simboli globalmente
mx, my, x, y, theta, v, w, dt = symbols('mx my x y theta v w dt')

# Parametri di rumore predefiniti
a = [0.1, 0.01, 0.01, 0.1] 

def get_symbolic_functions():
    """
    Genera le funzioni Python dai modelli simbolici.
    Queste funzioni contengono le singolarità (1/w) e verranno chiamate
    solo quando w non è vicino a zero.
    """
    # --- MODELLO DI MOTO GENERALE (Moto Curvilineo) ---
    R = v / w
    beta = theta + w * dt # Beta variabile ausiliaria
    
    # g(u, x) simbolica
    gux_sym = Matrix([
        x - R * sympy.sin(theta) + R * sympy.sin(beta),
        y + R * sympy.cos(theta) - R * sympy.cos(beta),
        theta + w * dt
    ])

    # Jacobiani Simbolici
    Gt_sym = gux_sym.jacobian(Matrix([x, y, theta])) # Jacobiano rispetto allo Stato
    Vt_sym = gux_sym.jacobian(Matrix([v, w])) # Jacobiano rispetto al Comando

    # --- MODELLO DI MISURA (Landmark) ---
    # Range e Bearing
    hx_sym = Matrix([
        sympy.sqrt((mx - x) ** 2 + (my - y) ** 2),
        sympy.atan2(my - y, mx - x) - theta
    ])
    
    Ht_sym = hx_sym.jacobian(Matrix([x, y, theta])) # Jacobiano rispetto allo Stato

    # --- LAMBDIFY ---
    # Generiamo le funzioni "grezze" che calcolano le formule esatte
    
    # Gt e Vt accettano (x, y, theta, v, w, dt)
    _raw_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt_sym, "numpy")
    _raw_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt_sym, "numpy")
    
    # Ht accetta (x, y, theta, mx, my)
    # Ht non ha problemi di singolarità con w, quindi lo usiamo direttamente
    eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht_sym, "numpy")

    return _raw_Gt, _raw_Vt, eval_Ht

# Generiamo le funzioni "grezze" all'avvio
_raw_Gt, _raw_Vt, eval_Ht = get_symbolic_functions()
# Gt e Vt sono raw perché contengono singolarità (1/w)


# ------------------------------------------------------------------------------
# WRAPPER CON GESTIONE SINGOLARITÀ (w ~= 0)
# ------------------------------------------------------------------------------

def eval_gux(mu, u,sigma_u, dt):
    """
    Velocity Motion Model.
    Calcola la predizione dello stato (Media).
    Gestisce il caso w=0.
    """



    # Estrai variabili dallo stato e dal comando
    x_val, y_val, theta_val = mu
    v_val, w_val = u
    
    if abs(w_val) < 1e-6:
        # Modello Rettilineo (Si ottiene eseguendo il limite per w->0)
        # x' = x + v*dt*cos(theta)
        # y' = y + v*dt*sin(theta)
        # theta' = theta
        x_new = x_val + v_val * dt * np.cos(theta_val)
        y_new = y_val + v_val * dt * np.sin(theta_val)
        theta_new = theta_val + w_val * dt # w piccolissimo ma lo teniamo
    else:
        # Modello Curvilineo (Standard)
        r = v_val / w_val
        x_new = x_val - r * np.sin(theta_val) + r * np.sin(theta_val + w_val * dt)
        y_new = y_val + r * np.cos(theta_val) - r * np.cos(theta_val + w_val * dt)
        theta_new = theta_val + w_val * dt

    return np.array([x_new, y_new, theta_new])


def eval_Gt(x_val, y_val, theta_val, v_val, w_val, dt_val):
    """
    Jacobiano del Moto rispetto allo Stato Gt.
    Gestisce il caso w=0.
    """
    if abs(w_val) < 1e-6:
        # Jacobiano del moto rettilineo, ottenuto eseguendo il limite per w->0:
        # x' = x + v*dt*cos(theta) -> dx'/dtheta = -v*dt*sin(theta)
        # y' = y + v*dt*sin(theta) -> dy'/dtheta =  v*dt*cos(theta)
        ss = np.sin(theta_val)
        cc = np.cos(theta_val)
        return np.array([
            [1.0, 0.0, -v_val * dt_val * ss],
            [0.0, 1.0,  v_val * dt_val * cc],
            [0.0, 0.0, 1.0]
        ])
    else:
        # Usa la formula complessa generata da Sympy
        return _raw_Gt(x_val, y_val, theta_val, v_val, w_val, dt_val)
    
def landmark_model_hx(x, y, theta, mx, my):
        # Funzione h(x) standard 3D
        dx = mx - x
        dy = my - y
        r = math.sqrt(dx**2 + dy**2)
        phi = math.atan2(dy, dx) - theta
        phi = math.atan2(math.sin(phi), math.cos(phi))
        return np.array([r, phi])

def eval_Vt(x_val, y_val, theta_val, v_val, w_val, dt_val):
    """
    Jacobiano del Moto rispetto al Comando Vt.
    Gestisce il caso w=0.
    """
    if abs(w_val) < 1e-6:
        # Jacobiano del moto rettilineo rispetto a (v, w), ottenuto eseguendo il limite per w->0:
        # dx'/dv = dt*cos(theta)
        # dy'/dv = dt*sin(theta)
        # dtheta'/dw = dt
        ss = np.sin(theta_val)
        cc = np.cos(theta_val)
        
        # Approssimazione lineare:
        # Colonna 1 (derivata rispetto a v)
        # Colonna 2 (derivata rispetto a w -> approssimiamo a 0 o termini piccoli del 2° ordine)
        return np.array([
            [dt_val * cc, 0.0],
            [dt_val * ss, 0.0],
            [0.0,         dt_val]
        ])
    else:
        # Usa la formula complessa generata da Sympy
        return _raw_Vt(x_val, y_val, theta_val, v_val, w_val, dt_val)


# ------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------

# def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
#     # Funzione dummy o usata per il sampling
#     pass