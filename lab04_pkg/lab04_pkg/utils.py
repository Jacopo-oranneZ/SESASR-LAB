import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from math import atan2
from numpy import linalg as la

###############################################
# From EKF utils.py
###############################################

def residual(a, b, **kwargs):
    """
    Compute the residual between expected and sensor measurements, normalizing angles between [-pi, pi)
    If passed, angle_indx should indicate the positional index of the angle in the measurement arrays a and b

    Returns:
        y [np.array] : the residual between the two states
    """
    y = a - b

    if 'angle_idx' in kwargs:
        angle_idx = kwargs["angle_idx"]
        theta = y[angle_idx]
        y[angle_idx] = normalize_angle(theta)
        
    return y


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    if len(actual.shape)==1 and len(predicted.shape)==1:
        return np.mean(np.square(_error(actual, predicted)), axis=0)
    return np.mean(np.sum(np.square(_error(actual, predicted)), axis=1), axis=0)

def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))

def mae(error: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(error))

def normalize(arr: np.ndarray):
    """ normalize vector for plots """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def normalize_angle(theta):
    """
    Normalize angles between [-pi, pi)
    """
    theta = theta % (2 * np.pi)  # force in range [0, 2 pi)
    if theta > np.pi:  # move to [-pi, pi)
        theta -= 2 * np.pi
    
    return theta

def state_mean(sigmas, Wm):
    x = np.zeros(3)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))
    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))
    x[2] = atan2(sum_sin, sum_cos)
    return x

def z_mean(sigmas, Wm):
    dim_z = sigmas.shape[1]
    x = np.zeros(dim_z)

    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 1]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 1]), Wm))

    x[0] = np.sum(np.dot(sigmas[:,0], Wm))
    x[1] = atan2(sum_sin, sum_cos)
    return x

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

###############################################
# From Sensor Models utils.py
###############################################

def normalize_angle(theta):
    """
    Normalize angles between [-pi, pi)
    """
    theta = theta % (2 * np.pi)  # force in range [0, 2 pi)
    if np.isscalar(theta):
        if theta > np.pi:  # move to [-pi, pi)
            theta -= 2 * np.pi
    else:
        theta_ = theta.copy()
        theta_[theta>np.pi] -= 2 * np.pi
        return theta_
    
    return theta

# Exponential Function
def exponential(x, _lambda):
    return _lambda * np.exp(-1 * _lambda * x)

# Gaussian Function
def gaussian(x, mu, sigma):
    return (1.0 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)

# Normalized Gaussian pdf
def compute_p_hit_dist(dist, max_dist, sigma):
    '''
    Compute the hit probability p_hit for a given distance measurement.
    Args:
        dist: observed distance measurement
        max_dist: maximum measurable distance
        sigma: standard deviation of the Gaussian noise
    Returns:
        p_hit: normalized hit probability
    '''
    # Normalize the Gaussian over [0, max_dist]
    normalize_hit = 1e-9
    for j in range(round(max_dist)):
        normalize_hit += gaussian(j, 0., sigma)
    normalize_hit = 1. / normalize_hit

    p_hit = gaussian(dist, 0., sigma)*normalize_hit

    return p_hit

# Vectorized probabilistic beam model
def precompute_p_hit_map(distances, max_dist=None, sigma=1.0):
    """
    Vectorized probabilistic beam model.
    Supports scalar or ndarray inputs for distances (must be broadcastable).
    
    Args:
        dist:     observed distances measurement(s), ndarray or float
        z_max:    max range (scalar)
        sigma:    std dev for hit Gaussian

    Returns:
        p_hit (hit prob)
    """
    dist_flat = np.asarray(distances.ravel())

    if max_dist is None:
        max_dist = np.max(dist_flat)
    
    # --- Hit mode normalization ---
    j = np.arange(int(max_dist))
    normalize_hit = np.sum(gaussian(j[:, None], dist_flat, sigma), axis=0)
    normalize_hit = normalize_hit.reshape(distances.shape)
    normalize_hit = np.where(normalize_hit > 0, 1.0 / normalize_hit, 1.0)

    # Hit probability
    p_hit = gaussian(distances, np.zeros_like(distances), sigma) * normalize_hit
    return p_hit

# Vectorized probabilistic beam model
def evaluate_range_beam_dist_array(z, z_star, z_max, _mix_density, _sigma, _lamb_short):
    """
    Vectorized probabilistic beam model.
    Supports scalar or ndarray inputs for z and z_star (must be broadcastable).
    
    Args:
        z:        observed measurement(s), ndarray or float
        z_star:   expected measurement(s), ndarray or float
        z_max:    max range (scalar)
        _mix_density: mixture weights [z_hit, z_short, z_max, z_rand]
        _sigma:   std dev for hit Gaussian
        _lamb_short: lambda for short distribution

    Returns:
        p_hit, p_short, p_max, p_rand, p (stacked array), p_z (mixture prob)
        Shapes follow broadcast of z and z_star
    """

    z = np.asarray(z)
    z_star = np.asarray(z_star)

    # --- Hit mode normalization ---
    j = np.arange(int(z_max))
    normalize_hit = np.sum(gaussian(j[:, None], z_star.ravel(), _sigma), axis=0)
    normalize_hit = normalize_hit.reshape(z_star.shape)
    normalize_hit = np.where(normalize_hit > 0, 1.0 / normalize_hit, 1.0)

    # Hit probability
    p_hit = gaussian(z, z_star, _sigma) * normalize_hit

    # --- Short mode ---
    normalize_short = 1.0 - np.exp(-_lamb_short * z_star)
    p_short = np.where(
        z <= z_star,
        _lamb_short * np.exp(-_lamb_short * z) / (normalize_short + 1e-12),
        0.0
    )

    # --- Max mode ---
    p_max = np.where(z == z_max, 1.0, 0.0)

    # --- Random mode ---
    p_rand = np.full_like(z, 1.0 / z_max, dtype=float)

    # Stack all components: shape (..., 4)
    p = np.stack([p_hit, p_short, p_max, p_rand], axis=-1)

    # Weighted mixture probability
    p_z = np.tensordot(p, _mix_density, axes=([-1],[0]))

    return p_hit, p_short, p_max, p_rand, p_z

# Plot the distribution of z samples
def plot_sampling_dist(samples, title="Distribution of z samples", fig_name="z_star_hist.pdf"):
    '''
    Plot the distribution of z samples.
    Args:
        samples: array of z samples
        title: title of the plot
        fig_name: name of the file to save the plot
    '''
    
    n_bins = 100
    plt.hist(samples, n_bins)
    plt.title(title)
    plt.grid()
    plt.savefig(fig_name)
    plt.show()
    plt.close('all')


# Bresenham's line algorithm https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
def bresenham_v0(x0, y0, x1, y1):
    ret = []

    dx =  abs(x1-x0)
    sx = 1 if (x0<x1) else -1
    dy = -abs(y1-y0)
    sy = 1 if (y0<y1) else -1
    err = dx+dy

    while (True):
        ret.append((x0, y0))
        if (x0==x1 and y0==y1):
            break
        e2 = 2*err
        if (e2 >= dy):
            err += dy
            x0 += sx
        if (e2 <= dx):
            err += dx
            y0 += sy

    return ret


def bresenham(x0, y0, x1, y1, map):
    """""
    x0, y0: coordinate of the starting point (robot position)
    x1, y1: map coordinate of the max range point
    map: 2D occupancy grid map
    return: coordinate of the obstacle point or the map boundary point
    """""

    dx =  abs(x1 - x0)
    sx = 1 if (x0 < x1) else -1
    dy = -abs(y1 - y0)
    sy = 1 if (y0 < y1) else -1
    err = dx + dy

    while (True):
        # check if obstacle encountered or ray reach end of map
        if x0 < 0.:
            obst = 0, y0
            break
        elif x0 >= map.shape[0]: # check if map border reached
            obst = x0, y0
            break
        elif y0 < 0.:
            obst = x0, 0
            break
        elif y0 >= map.shape[1]:
            obst = x0, y0
            break
        # elif map[int(x0-1), int(y0-1)]==1 or map[int(x0), int(y0)]==1 or map[int(x0-1), int(y0)]==1 or map[int(x0), int(y0-1)]==1 or ((x0==x1) and (y0==y1)):
        #     obst = [x0, y0]
        #     print("obst", obst)
        #     break
        elif map[int(x0), int(y0)]==1 or ((x0==x1) and (y0==y1)):
            obst = [x0, y0]
            break

        e2 = 2*err
        if (e2 >= dy):
            err += dy
            x0 += sx
        if (e2 <= dx):
            err += dx
            y0 += sy
        
    return obst
