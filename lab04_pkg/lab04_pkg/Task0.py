import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy

from scipy.stats import norm
from  matplotlib.patches import Arc
from lab04_pkg.utils import compute_p_hit_dist
import sympy
sympy.init_printing(use_latex='mathjax')
from sympy import  Matrix, symbols
from math import cos, sin, degrees

eval_Ht = None
landmarks = None
arrow = u'$\u2191$'

################################
### Motion model functions #####
################################
def sample_normal_distribution(sigma_sqrd):
    return 0.5 * np.sum(np.random.default_rng().uniform(-np.sqrt(sigma_sqrd), np.sqrt(sigma_sqrd), 12))

def evaluate_sampling_dist(mu, sigma, n_samples_mot, sample_function):
    n_bins = 100
    samples = []

    for i in range(n_samples_mot):
        samples.append(sample_function(mu, sigma))

    print("%30s : mean = %.3f, std_dev = %.3f" % ("Normal", np.mean(samples), np.std(samples)))

    count, bins, ignored = plt.hist(samples, n_bins)
    plt.plot(bins, norm(mu, sigma).pdf(bins), linewidth=2, color='r')
    plt.xlim([mu - 5*sigma, mu + 5*sigma])
    plt.title("Normal distribution of samples")
    plt.grid()
    plt.savefig("gaussian_dist.pdf")
    plt.show()

def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    a -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6]
    dt -- time interval of prediction
    """
    v_hat = u[0] + np.random.normal(0, a[0]*u[0]**2 + a[1]*u[1]**2)
    w_hat = u[1] + np.random.normal(0, a[2]*u[0]**2 + a[3]*u[1]**2)
    gamma_hat = np.random.normal(0, a[4]*u[0]**2 + a[5]*u[1]**2)

    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime])

def plot_graph(a,u,dt,n_samples_mot,x):
     
    x_prime = np.zeros([n_samples_mot, 3])
    for i in range(n_samples_mot):
        x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)

    ###################################
    ######### Plot x samples ##########
    ###################################
       
    mu = np.mean(x_prime, axis=0)
    sigma = np.std(x_prime, axis=0)
    evaluate_sampling_dist(mu[0], sigma[0], n_samples_mot, np.random.normal)

    ###################################
    ### Sampling the velocity model ###
    ###################################

    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x[2])-90)
    plt.scatter(x[0], x[1], marker=rotated_marker, s=100, facecolors='none', edgecolors='b')

    for x_ in x_prime[:200]:
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(degrees(x_[2])-90)
        plt.scatter(x_[0], x_[1], marker=rotated_marker, s=40, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("velocity motion model sampling")
    plt.savefig("velocity_samples.pdf")
    plt.show()

    ###################################
    #### Multiple steps of sampling ###
    ###################################

    cmds = [
        [0.8, 0],
        [0.8, 0.0],
        [0.6, 0.5],
        [0.6, 0.5],
        [0.6, 1.5],
        [0.6, 0],
        [0.8, 0.0],
        [0.7, -0.5],
        [0.7, -0.5],
        [0.5, -1.5],
        [0.8, 0],
        [0.8, 0.0]
    ]

    x_prime = np.zeros([n_samples_mot, 3])
    for t, u in enumerate(cmds):
        for i in range(0, n_samples_mot):
            x_ = x_prime[i,:]
            if t ==0:
                x_prime[i,:] = sample_velocity_motion_model(x, u, a, dt)
            else:
                x_prime[i,:] = sample_velocity_motion_model(x_, u, a, dt)
        
        plt.plot(x_prime[:,0], x_prime[:,1], "r,")
        plt.plot(x[0], x[1], "bo")

        x = np.mean(x_prime, axis=0)
        sigma = np.std(x_prime, axis=0)
        print("mu: ", x, "sigma: ", sigma)
    
    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("velocity multiple sampling")
    plt.savefig("multi_velocity_samples.pdf")
    plt.show()

    plt.close('all')

def compute_jacobian():
    mx, my, x, y, theta, v, w, dt = symbols('mx my x y theta v w dt')
    R = v / w
    beta = theta + w * dt
    gux = Matrix(
    [
        [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
        [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
        [beta],
    ]
   )

    hx = Matrix(
        [
            [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)], # range
            [sympy.atan2(my - y, mx - x) - theta],       # bearing
        ]
    )

    #eval_gux = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')
    Gt = gux.jacobian(Matrix([x, y, theta]))
    eval_Gt = sympy.lambdify((x, y, theta, v, w, dt), Gt, "numpy")
    Vt = gux.jacobian(Matrix([v, w]))
    eval_Vt = sympy.lambdify((x, y, theta, v, w, dt), Vt, "numpy")


    Ht = hx.jacobian(Matrix([x, y, theta]))
    eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, "numpy")

    return eval_Gt, eval_Vt, eval_Ht

################################
### Landmark model functions ###
################################

def landmark_range_bearing_sensor(robot_pose, landmark, sigma, max_range=6.0, fov=math.pi/2):
    """""
    Simulate the detection of a landmark with a virtual sensor able to estimate range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]

    r_ = math.dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = math.atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])

    # filter z for a more realistic sensor simulation (add a max range distance and a FOV)
    if r_ > max_range or abs(phi_) > fov / 2:
        return None

    return [r_, phi_]

def landmark_model_prob(z, landmark, robot_pose, max_range, fov, sigma):
    """""
    Landmark sensor model algorithm:
    Inputs:
      - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
      - landmark: the landmark position in the map [m_x, m_y]
      - x: the robot pose [x,y,theta]
    Outputs:
     - p: the probability p(z|x,m) to obtain the measurement z from the state x
        according to the estimated range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta = robot_pose[:]
    sigma_r, sigma_phi = sigma[:]

    r_hat = math.dist([x, y], [m_x, m_y])
    phi_hat = math.atan2(m_y - y, m_x - x) - theta
    p = compute_p_hit_dist(z[0] - r_hat, max_range, sigma_r) * compute_p_hit_dist(z[1] - phi_hat, fov/2, sigma_phi)

    return p

def landmark_model_sample_pose(z, landmark, sigma):
    """""
    Sample a robot pose from the landmark model
    Inputs:
        - z: the measurements features (range and bearing of the landmark from the sensor) [r, phi]
        - landmark: the landmark position in the map [m_x, m_y]
        - sigma: the standard deviation of the measurement noise [sigma_r, sigma_phi]
    Outputs:
        - x': the sampled robot pose [x', y', theta']
    """""
    m_x, m_y = landmark[:]
    sigma_r, sigma_phi = sigma[:]

    gamma_hat = np.random.uniform(0, 2*math.pi)
    r_hat = z[0] + np.random.normal(0, sigma_r)
    phi_hat = z[1] + np.random.normal(0, sigma_phi)

    x_ = m_x + r_hat * math.cos(gamma_hat)
    y_ = m_y + r_hat * math.sin(gamma_hat)
    theta_ = gamma_hat - math.pi - phi_hat

    return np.array([x_, y_, theta_])

def plot_sampled_poses(robot_pose, z, landmark, sigma,n_samples_sens):  
    """""
    Plot sampled poses from the landmark model
    """""
    # plot samples poses
    for i in range(n_samples_sens):
        x_prime = landmark_model_sample_pose(z, landmark, sigma)
        # plot robot pose
        rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
        rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(x_prime[2])-90)
        plt.scatter(x_prime[0], x_prime[1], marker=rotated_marker, s=80, facecolors='none', edgecolors='b')
    
    # plot real pose
    rotated_marker = mpl.markers.MarkerStyle(marker=arrow)
    rotated_marker._transform = rotated_marker.get_transform().rotate_deg(math.degrees(robot_pose[2])-90)
    plt.scatter(robot_pose[0], robot_pose[1], marker=rotated_marker, s=140, facecolors='none', edgecolors='r')

    plt.xlabel("x-position [m]")
    plt.ylabel("y-position [m]")
    plt.title("Landmark Model Pose Sampling")
    # plt.savefig("landmark_model_sampling.pdf")
    plt.show()

def plot_landmarks(landmarks, robot_pose, z, p_z, max_range=6.0, fov=math.pi/4):
    """""
    Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    """""
    x, y, theta = robot_pose[:]

    start_angle = theta + fov/2
    end_angle = theta - fov/2

    plt.figure()
    ax = plt.gca()
    # plot robot pose
    # find the virtual end point for orientation
    endx = x + 0.5 * math.cos(theta)
    endy = y + 0.5 * math.sin(theta)
    plt.plot(x, y, 'or', ms=10)
    plt.plot([x, endx], [y, endy], linewidth = '2', color='r')

    # plot FOV
    # get ray target coordinates
    fov_x_left = x + math.cos(start_angle) * max_range
    fov_y_left = y + math.sin(start_angle) * max_range
    fov_x_right = x + math.cos(end_angle) * max_range
    fov_y_right = y + math.sin(end_angle) * max_range

    plt.plot([x, fov_x_left], [y, fov_y_left], linewidth = '1', color='b')
    plt.plot([x, fov_x_right], [y, fov_y_right], linewidth = '1', color='b')

    R = max_range
    a, b = 2*R, 2*R
    arc = Arc((x, y), a, b,
                 theta1=math.degrees(end_angle), theta2=math.degrees(start_angle), color='b', lw=1.2)
    ax.add_patch(arc)

    # plot landmarks
    for i, lm in enumerate(landmarks):
        plt.plot(lm[0], lm[1], "sk", ms=10, alpha=0.7)

    # plot perceived landmarks position and associated probability (color scale)
    lm_z = np.zeros((len(z), 2))
    for i in range(len(z)):
        # draw endpoint with probability from Likelihood Fields
        lx = x + z[i][0] * math.cos(z[i][1]+theta)
        ly = y + z[i][0] * math.sin(z[i][1]+theta)
        lm_z[i, :] = lx, ly
    
    col = np.array(p_z)
    plt.scatter(lm_z[:,0], lm_z[:,1], s=60, c=col, cmap='viridis')
    plt.colorbar()

    plt.show()
    plt.close('all')


    mx, my, x, y, theta = symbols("m_x m_y x y theta")
    hx = Matrix(
        [
            [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)], # range
            [sympy.atan2(my - y, mx - x) - theta],       # bearing
        ]
    )
    # eval_hx = sympy.lambdify((x, y, theta, mx, my), hx, "numpy")

    global eval_Ht
    Ht = hx.jacobian(Matrix([x, y, theta]))
    eval_Ht = sympy.lambdify((x, y, theta, mx, my), Ht, "numpy")

    return eval_Ht

def main():
    ##############################
    ### Motion model example ###
    ##############################
    n_samples_mot = 500
    n_bins = 100
    dt = 0.5

    x = [2, 4, 0]
    u = [0.8, 0.6]
    a_w = [0.001, 0.01, 0.1, 0.2, 0.05, 0.05] # noise variance
    a_v = [0.05, 0.09, 0.002, 0.01, 0.05, 0.05] # noise variance

    plot_graph(a_w, u, dt, n_samples_mot,x)
    plot_graph(a_v, u, dt, n_samples_mot,x)
     

    ##############################
    ### Landmark model example ###
    ##############################

    n_samples_sens = 1000
    # robot pose
    robot_pose = np.array([0., 0., math.pi/4])
    # landmarks position in the map
    global landmarks
    landmarks = [
                 np.array([5., 2.]),
                 np.array([-2.5, 3.]),
                 np.array([3., 1.5]),
                 np.array([4., -1.]),
                 np.array([-2., -2.])
                 ]
    # sensor parameters
    fov = math.pi/3 # field of view
    max_range = 6.0 # max range distance
    sigma = np.array([0.3, math.pi/24]) # range and bearing noise standard deviation

    # compute measurements and associated probability
    z = []
    p = []
    for i in range(len(landmarks)):
        # read sensor measurements (range, bearing)
        z_i = landmark_range_bearing_sensor(robot_pose, landmarks[i], sigma=sigma, max_range=max_range, fov=fov)
         
        if z_i is not None: # if landmark is not detected, the measurement is None
            z.append(z_i)
            # compute the probability for each measurement according to the landmark model algorithm
            p_i = landmark_model_prob(z_i, landmarks[i], robot_pose, max_range, fov, sigma)
            p.append(p_i)

    print("Probability density value:", p)
    # Plot landmarks, robot pose with sensor FOV, and detected landmarks with associated probability
    plot_landmarks(landmarks, robot_pose, z, p, fov=fov)

    ##########################################
    ### Sampling poses from landmark model ###
    ##########################################
    if len(z) == 0:
        print("No landmarks detected!")
        return
    
    # consider only the first landmark detected
    landmark = landmarks[0]
    z = landmark_range_bearing_sensor(robot_pose, landmark, sigma)

    # plot landmark
    plt.plot(landmark[0], landmark[1], "sk", ms=10)
    plot_sampled_poses(robot_pose, z, landmark, sigma,n_samples_sens)
    
    [Gt_sym, Vt_sym, Ht_sym] = compute_jacobian()
    Gt = Gt_sym(x[0], x[1], x[2], u[0], u[1], dt)
    Vt = Vt_sym(x[0], x[1], x[2], u[0], u[1], dt)
    Ht = Ht_sym(robot_pose[0], robot_pose[1], robot_pose[2], landmark[0], landmark[1])
    print("Jacobian Gt:\n", Gt)
    print("Jacobian Vt:\n", Vt)
    print("Jacobian Ht:\n", Ht)

    plt.close('all')

if __name__ == "__main__":
    main()