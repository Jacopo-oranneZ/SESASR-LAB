import numpy as np
from lab04_pkg.Task0 import eval_gux
mu=[2.0, 2.0, 0.0]
alfas = [0.1, 0.1, 0.05, 0.05]  # Motion noise parameters
commands = [1.0, 0.2]  # [v, w]
dt = 0.1  # Time step

mu_noise = mu + np.random.randn(3) * [0.1, 0.1, 0.005]
for i in range(500):
    eval_gux(mu_noise, commands, alfas, dt)