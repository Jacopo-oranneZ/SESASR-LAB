import numpy as np
from numpy.linalg import inv

class RobotEKF:
    def __init__(
        self,
        dim_x=1,
        dim_u=1,
        eval_gux=None,
        eval_hx=None,
        eval_Gt=None,
        eval_Vt=None,
        eval_Ht=None,
    ):
        """
        Initializes the extended Kalman filter creating the necessary matrices
        """
        self.mu = np.zeros((dim_x))  # mean state estimate
        self.Sigma = np.eye(dim_x)   # covariance state estimate
        self.Mt = np.eye(dim_u)      # process noise covariance matrix

        # Functions for motion and measurement models and their Jacobians
        self.eval_gux = eval_gux
        self.eval_hx = eval_hx
        self.eval_Gt = eval_Gt
        self.eval_Vt = eval_Vt
        self.eval_Ht = eval_Ht

        self._I = np.eye(dim_x)  # identity matrix used for computations

    def predict(self, u, sigma_u, g_extra_args=()):
        """
        Update the state prediction using the control input u and compute the relative uncertainty ellipse
        """
        
        # 1. AGGIORNAMENTO Mt (Process Noise Covariance)
        # sigma_u contiene le varianze calcolate nel nodo (alpha_1*v^2 + ...)
        # Dobbiamo aggiornare la matrice Mt diagonale con questi valori attuali.
        if sigma_u is not None and len(sigma_u) > 0:
            self.Mt = np.diag(sigma_u)

        # 2. Predizione dello stato (Media)
        # eval_gux usa u (comando) e mu (stato prec), sigma_u qui è ignorato per la media
        self.mu = self.eval_gux(self.mu, u, sigma_u, *g_extra_args)

        # 3. Predizione della Covarianza (Sigma)
        args = (*self.mu,) if u is None else (*self.mu, *u)
        
        # Calcolo Jacobiani
        Gt = self.eval_Gt(*args, *g_extra_args)
        Vt = self.eval_Vt(*args, *g_extra_args)
        
        # Propagazione incertezza: P_bar = G*P*G' + V*M*V'
        # Qui self.Mt è fondamentale che sia aggiornato!
        self.Sigma = Gt @ self.Sigma @ Gt.T + Vt @ self.Mt @ Vt.T

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(), residual=np.subtract, **kwargs):
        """Performs the update innovation of the extended Kalman filter."""

        # Convert the measurement to a vector if necessary
        if np.isscalar(z):
            z = np.asarray([z], float)
            Qt = np.atleast_2d(Qt).astype(float)
            if not isinstance(Ht_args, tuple): # Safety check
                 Ht = np.atleast_2d(eval_Ht).astype(float)
        
        # Compute the Jacobian Ht

        Ht = eval_Ht(*Ht_args)
        
        # Assicuriamoci che Ht sia 2D (per misure scalari come IMU)
        if Ht.ndim == 1:
            Ht = Ht[np.newaxis, :]

        # 1. Kalman Gain
        # S = H*P*H' + Q
        SigmaHT = self.Sigma @ Ht.T
        self.S = Ht @ SigmaHT + Qt
        
        # K = P*H'*inv(S)
        self.K = SigmaHT @ inv(self.S)

        # 2. Innovation (Residual) y = z - h(x)
        z_hat = eval_hx(*hx_args)
        if np.isscalar(z_hat):
            z_hat = np.asarray([z_hat], float)

        y = residual(z, z_hat, **kwargs)
        
        # 3. State Update: x = x + K*y
        self.mu = self.mu + self.K @ y

        # 4. Covariance Update: P = (I - K*H)*P
        # Usiamo la forma numericamente più stabile (Joseph form) se possibile, 
        # ma quella standard va bene per questo lab: P = (I-KH)P
        I_KH = self._I - self.K @ Ht
        
        # Forma semplice (spesso sufficiente)
        # self.Sigma = I_KH @ self.Sigma
        
        # Forma stabile (Joseph form) simmetrica: P = (I-KH)P(I-KH)' + KQK'
        # Garantisce che la matrice rimanga definita positiva
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T