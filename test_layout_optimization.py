
"""
Implemented:
    - Continuous wake model
    -     
"""

##################################################################################### NOTES 
"""
1/12-22: 'power_ind_turbine': error in the implementation of power calculation for U > U_cut_out:
              U = U_cut_out see Eq.(3) in [1]
              Also an error in U_cut_in < U < U_cut_out case
              

          
Refs:
[1] J.Partk , K.H. Law/Applied Energy 151 (2015)
"""
#====================================================================================


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, LinearConstraint
import functools
from matplotlib import cm


######################################################################################## Decorators 
# -------------------------------------------------------------------------------------- cost function decor
def rectangular_area(xlim, ylim):
    """
    Decorator that adds a cost for transgressing a predescribed rectangular area
    Args:
        xlim: [x_min, x_max], where x_min and x_max are the boundaries of the  
            rectangle in the x-direction
        ylim: [y_min, y_max], where y_min and y_max are the boundaries of the  
            rectangle in the y-direction
    Out:
        Adds a penalty for going outside the prescribed rectangular domain. 
        The "severity" of the penalty is given by beta.
    """
    beta = 0.1
    def rectangular_area_inner(func):
        def wrapper(*args, **kwargs):
            x_min, x_max = xlim[0], xlim[1]
            y_min, y_max = ylim[0], ylim[1]
            penalty = 0.0
            for i in np.arange(0, x_vector.size, 2):
                x, y = x_vector[i], x_vector[i+1]
                if x < x_min:
                    penalty += np.exp(beta*(x_min-x)**2)-1
                elif x > x_max:
                    penalty += np.exp(beta*(x-x_max)**2)-1
                if y < y_min:
                    penalty += np.exp(beta*(y_min-y)**2)-1
                elif y > y_max:
                    penalty += np.exp(beta*(y-y_max)**2)-1
            return (1 - penalty)*func(*args, **kwargs)
        return wrapper
    return rectangular_area_inner

######################################################################################## 


def print_wind_field(x_all, U, wind_dir, D, plot_title="wind speed"):
    N_x = 240
    N_y = 240
    x_grid = np.linspace(-120,120,N_x)
    y_grid = np.linspace(-120,120,N_y)

    #x_grid = np.linspace(-50,100,N_x)
    #y_grid = np.linspace(-50,300,N_y)

    u_eval = np.zeros((N_x, N_y))
    delta_u_eval = np.zeros((N_turb, N_x, N_y))

    u_eval_optim = np.zeros((N_x, N_y))
    delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))


    for ix in range(N_x):
        for iy in range(N_y):
            x_i = (x_grid[ix], y_grid[iy])
            u_eval[ix,iy], temp = wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D, r_i = 0, theta_i = 0)
            delta_u_eval[:,ix,iy] = temp.flatten()

    fig = plt.figure(constrained_layout=True)

    (xv, yv) = np.meshgrid(x_grid, y_grid)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv, yv, u_eval.T, cmap=plt.cm.coolwarm)
    cset = ax.contourf(xv, yv, u_eval.T, zdir='z', offset=25, cmap=cm.coolwarm)
    ax.scatter3D(x_all[:,0], x_all[:,1], np.full((1,N_turb),25),'+k', s=4)


    plt.colorbar(surf, shrink=0.5) # an example

    fig.colorbar(surf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(plot_title)
    ax.azim = -90
    #ax.dist = 10
    ax.elev = 90    

def power_ind_turbine(U, U_cut_in, U_cut_out, C_p, rho, R0):
    """
    Compute power output of individual turbine
    Args:
        U: ambient/effective wind speed, scalar value
        U_cut_in: min speed for power production
        U_cut_out: max effective speed, if larger the yaw is assumed modified to obtain U=U_cut_out
        C_p: power coefficient (constant)
        rho: density of air
        R0: rotor radius
    Output:
        Power: the power of the individual turbine [what unit? what time horizon, annually?]
    """
    A = np.pi*R0**2
    if U < U_cut_in:
        Power = 0
    elif U > U_cut_out:
        Power = 0.5*rho*A*C_p*U_cut_out**3
    else:
        Power = 0.5*rho*A*C_p*U**3
        
    return Power
     
def wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D, r_i, theta_i):
    """
    Compute wind speed at downstream turbine i due to the wake of all upstream turbines
    Args:
        x_i: target location (does not need to be a windturbine location)
        x_all: location coordinates of all turbines
        U: freestream wind speed
        wind_dir: wind direction
        D: diameters (of upstream) turbines
    Output:
        u_i: wind speed at turbine i caused by all other turbines
        delta_u_i: wind deficit factors, turbine by turbine
    """

    N_theta = len(np.atleast_1d(theta_i))
    N_r = len(np.atleast_1d(r_i))
    num_turbines, temp = np.shape(x_all)
    delta_u_i = np.zeros((num_turbines, N_theta, N_r))

    #print("wind_dir: ", wind_dir, "Target turbine coord: ", x_i)
    
    for j in range(num_turbines):
        x_j = x_all[j,:]
        #theta_ij = np.arctan((x_i[0]-x_j[0])/(x_i[1]-x_j[1]))
        theta_ij = np.arctan2((x_i[0]-x_j[0]),(x_i[1]-x_j[1]))
    
        Eucl_dist = np.sqrt((x_i[0]-x_j[0])**2 + (x_i[1]-x_j[1])**2)
        downstream_dist_ij = Eucl_dist*np.cos(abs(theta_ij - wind_dir))
        radial_dist_ij =  Eucl_dist*np.sin(abs(theta_ij - wind_dir))
 
        r = np.sqrt((radial_dist_ij - np.outer(np.cos(theta_i), r_i) )**2 + (np.outer(np.sin(theta_i), r_i))**2)
        
        #print("Turbine ", j, "d_ij: ", downstream_dist_ij, "r_ij: ", radial_dist_ij, "theta_ij: ", theta_ij)
        
        delta_u_i[j,:,:], _, _ = wake_model_continuous(downstream_dist_ij, r, alpha, D[j])
        if downstream_dist_ij  < 0:
            delta_u_i[j] = 0
        
    delta_u = np.sqrt(np.sum(delta_u_i**2, axis=0))

    
    u_i = U*(1-delta_u)
    
    #if downstream_dist_ij  >= 0:
    #    u_ij = U*(1-delta_u)
    #else:
    #    u_ij = U
    
    return u_i, delta_u_i

def wake_model_continuous(d, r, alpha, D, averaged=False):
    
    # Linear wake expansion
    kappa = 0.05
    R0 = 0.5*D
    R = R0 + kappa*d
    #deficit factor Eq. (9)
    delta_u = 2*alpha*(R0/R)**2*np.exp(-(r/R)**2)
    
    #Partial_delta_u_Partial_d = -4*alpha*R0**2*kappa*np.exp(-r**2/R**2)*(R**(-3)+r/R)
    Partial_delta_u_Partial_d = 4*alpha*R0**2*kappa/R**3*np.exp(-r**2/R**2)*(r**2/R**2-1)

    Partial_delta_u_Partial_r = -2*r/R**2*delta_u
        
    return delta_u, Partial_delta_u_Partial_d, Partial_delta_u_Partial_r
    
def Simpson_2D(x_start, x_end, y_start, y_end, N_int, fun):
    """
    Currently not used; found built-in Simpson's rule instead
    """
    delta_x = (x_end - x_start)/N_int
    delta_y = (y_end - y_start)/N_int
    w = np.empty((N_int+1,),int)
    w[0] = 1
    w[N_int] = 1
    w[1:N_int-1] = np.tile([4,2],int(N_int/2-1))
    w[N_int-1] = 4
    W = np.outer(w,w)
    return W
    
def averaged_wind_speed(x_i, x_all, U, wind_dir, D, R0):
    """
    Average the wind speed described by u_fun over the disc with radius R0 (actuator disc model?) using numerical integration in polar coordinates
    Args:
        x_i: coordinates of target wind turbine
        x_all: coordinates of all wind turbines
        U: ambient wind speed
        wind_dir: wind direction
        D: diameters of (all) wake-inducing wind turbines
        R0: radius of target turbine i;
        
    Output:
        u_averaged: average wind speed (scalar)
        delta_averaged: averaged wind speed reduction from all turbines (vector)
    """
    # Discretization of radius and angle for wind speed numerical integration over the rotor swept area (assumed circular)
    theta_min, theta_max, n_points_theta = (0, 2*np.pi, 30)
    r_min, r_max, n_points_r = (1e-2, R0, 20)
    theta_i = np.linspace(theta_min, theta_max, n_points_theta)
    r_i = np.linspace(r_min, r_max, n_points_r)

    # Get wake-reduced wind speed point-wise
    u_evals, delta_u_evals = wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D, r_i, theta_i)
    integrand_pts_u = u_evals*np.tile(r_i,(n_points_theta,1))
    integrand_pts_delta_u = delta_u_evals*np.tile(r_i,(n_points_theta,1))
   
    # Numerical integration Simpson's rule 2D, polar coordinates
    u_averaged = 1/(np.pi*R0**2)*simps(simps(integrand_pts_u, theta_i, axis = 0),r_i)

    delta_u_averaged = 1/(np.pi*R0**2)*simps(simps(integrand_pts_delta_u, theta_i, axis = 1),r_i, axis=1)
    
    return u_averaged, delta_u_averaged
    
# def objective_fun(x_vector, U, wind_dir, R0, alpha, rho, U_cut_in, U_cut_out, C_p):
#     """
#     Compute objective function and its gradient
    
#     Args:
#         x_all: locations of all turbines
#         U: ambient wind speed
#         wind_dir: wind direction
#         R0: rotor radius of turbines
        
#     Output:
#         P: objective function: total power production
#         g: gradient of P w.r.t. spatial coordinates of all turbines
#     """
    
#     #print("Checking x_vector: ", np.shape(x_vector))
#     #    np.atleast_1d(
#     N_turb = int(np.shape(x_vector)[0]/2)
#     x_all = np.reshape(x_vector, (N_turb, 2))
#     D = 2*R0
#     #N_turb, temp = np.shape(x_all)
    
    
#     Partial_u_i_Partial_x_q = np.zeros((N_turb,N_turb))
#     Partial_u_i_Partial_y_q = np.zeros((N_turb,N_turb))
                    
#     # Averaged wind and deficit factors
#     delta_u_ij = np.zeros((N_turb, N_turb))
#     u_i = np.zeros((N_turb,1))
#     p_i = np.zeros((N_turb,1))
        
#     for i in range(N_turb):
#         x_i = x_all[i,:]
#         u_i, delta_uij_temp = averaged_wind_speed(x_i, x_all, U, wind_dir, D, R0[i])
#         delta_u_ij[i,:] = delta_uij_temp.T
    
#         #delta_u_ij[i,i] = 1e-5
    
#         p_i[i] = power_ind_turbine(u_i, U_cut_in, U_cut_out, C_p, rho, R0[i])
        
#     P = np.sum(p_i)
    
#     #print("delta_u_ij: ", delta_u_ij)
    
#     # First factor of current (3):
#     Partial_u_i_Partial_delta_u_ik = np.zeros((N_turb,N_turb))
#     for i in range(N_turb):
#         for k in range(N_turb):
#             if k != i:
#                 if abs(delta_u_ij[i,k]) > 1e-6:
#                     Partial_u_i_Partial_delta_u_ik[i,k] = -U*np.reciprocal(np.sqrt(np.sum(np.square(delta_u_ij[i,:]))))*delta_u_ij[i,k]
#                     #Partial_u_i_Partial_delta_u_ik[i,k] = -U*np.reciprocal(np.sqrt(np.sum(np.square(delta_u_ij[:,i]))))*delta_u_ij[k,i]
                                
#                     #print("Partial_u_i_Partial_delta_u_ik[i,k]: ", Partial_u_i_Partial_delta_u_ik[i,k])
    
#     # Second factor of current (3):
#     Partial_delta_u_ik_Partial_x_q = np.zeros((N_turb,N_turb,N_turb))
#     Partial_delta_u_ik_Partial_y_q = np.zeros((N_turb,N_turb,N_turb))
    
        
    
    
#     # du_ik/dq
#     #Partial_delta_u_ik_Partial_q = np.zeros((N_turb, N_turb, N_turb))
    
#     # Discretization of radius and angle for wind speed numerical integration over the rotor swept area (assumed circular)
#     theta_min, theta_max, n_points_theta = (0, 2*np.pi, 30)
#     r_min, n_points_r = (1e-2, 20)
#     #r_min, r_max, n_points_r = (0, R0, 20)
    
#     theta_i = np.linspace(theta_min, theta_max, n_points_theta)
#     #r_i = np.linspace(r_min, r_max, n_points_r)

    
#     Partial_delta_u_Partial_d = np.zeros((N_turb, n_points_theta, n_points_r))
#     # First case, second factor D delta_u_ik/ D x_i, q=i
#     for i in range(N_turb):
#         x_i = x_all[i,:]
#         r_max = R0[i]
#         r_i = np.linspace(r_min, r_max, n_points_r)
#         for k in range(N_turb):
#             if k != i:
#                 x_k = x_all[k,:]
#                 theta_ik = np.arctan2((x_i[0]-x_k[0]),(x_i[1]-x_k[1]))
    
#                 Eucl_dist = np.sqrt((x_i[0]-x_k[0])**2 + (x_i[1]-x_k[1])**2)
#                 downstream_dist_ik = Eucl_dist*np.cos(abs(theta_ik - wind_dir))
#                 radial_dist_ik =  Eucl_dist*np.sin(abs(theta_ik - wind_dir))
                

#                 r = np.sqrt((radial_dist_ik - np.outer(np.cos(theta_i), r_i) )**2 + (np.outer(np.sin(theta_i), r_i))**2)
#                 d = downstream_dist_ik
        
                
#                 # Index k represents the turbine whose effect on turbine i is quantified, hence it should be R0[k] below [???? DOUBLE-CHECK!]
#                 _, Partial_delta_u_Partial_d, Partial_delta_u_Partial_r = wake_model_continuous(d, r, alpha, 2*R0[k])
                
                
#                 # Scalar expressions
#                 #Partial_d_Partial_x_i = ((x_i[1]-x_k[1])*np.sin(abs(wind_dir - theta_ik)) + (x_i[0]-x_k[0])*np.cos(abs(wind_dir - theta_ik)))/Eucl_dist
#                 Partial_d_Partial_x_i = np.sin(wind_dir) * np.sign(x_i[1]-x_k[1])
                
#                 #Partial_d_Partial_y_i = (-(x_i[0]-x_k[0])*np.sin(abs(wind_dir - theta_ik)) + (x_i[1]-x_k[1])*np.cos(abs(wind_dir - theta_ik)))/Eucl_dist
#                 Partial_d_Partial_y_i = np.cos(wind_dir) * np.sign(x_i[1]-x_k[1])
                
#                 # Matrix-values expressions
#                 #Partial_r_Partial_x_i = (Eucl_dist*np.sin(abs(wind_dir - theta_ik)) - np.outer(np.cos(theta_i), r_i))*(-Partial_d_Partial_y_i)/r
#                 Partial_r_Partial_x_i = (radial_dist_ik - np.outer(np.cos(theta_i), r_i))/r*np.cos(wind_dir) * np.sign(x_i[1]-x_k[1])*np.sign(theta_ik-wind_dir)
            
                
#                 Partial_r_Partial_y_i = -(radial_dist_ik - np.outer(np.cos(theta_i), r_i))/r*np.sin(wind_dir) * np.sign(x_i[1]-x_k[1])*np.sign(theta_ik-wind_dir)
                
                
#                 integrand_pts_xder = (Partial_delta_u_Partial_d*Partial_d_Partial_x_i + Partial_delta_u_Partial_r*Partial_r_Partial_x_i)*np.tile(r_i,(n_points_theta,1))
                                
#                 integrand_pts_yder = (Partial_delta_u_Partial_d*Partial_d_Partial_y_i + Partial_delta_u_Partial_r*Partial_r_Partial_y_i)*np.tile(r_i,(n_points_theta,1))
                
                
#                 # Numerical integration Simpson's rule 2D, polar coordinates
#                 # We assume integration around turbine i below, hence R0[i]; DOUBLE-CHECK!
#                 Partial_delta_u_ik_Partial_x_q[i,k,i] = 1/(np.pi*R0[i]**2)*simps( simps(integrand_pts_xder, theta_i, axis = 0),r_i)
                
#                 Partial_delta_u_ik_Partial_y_q[i,k,i] = 1/(np.pi*R0[i]**2)*simps( simps(integrand_pts_yder, theta_i, axis = 0),r_i)
                
#                 Partial_u_i_Partial_x_q[i,i] += Partial_u_i_Partial_delta_u_ik[i,k]*Partial_delta_u_ik_Partial_x_q[i,k,i]
#                 Partial_u_i_Partial_y_q[i,i] += Partial_u_i_Partial_delta_u_ik[i,k]*Partial_delta_u_ik_Partial_y_q[i,k,i]
                
#                 #print("First case, Partial_u_i_Partial_x_q[i,i]", Partial_u_i_Partial_x_q[i,i])
#                 #print("First case, Partial_u_i_Partial_y_q[i,i]", Partial_u_i_Partial_y_q[i,i])
                
                
#                 #print("Partial_delta_u_Partial_r: ", Partial_delta_u_Partial_r)
#                 #print("Partial_r_Partial_x_i: ", Partial_r_Partial_x_i)
                
#     # Second case, second factor D delta_u_iq/ D x_q
#     for i in range(N_turb):
#         x_i = x_all[i,:]
#         for q in range(N_turb):
#             if q != i:
#                 x_q = x_all[q,:]
#                 theta_iq = np.arctan2((x_i[0]-x_q[0]),(x_i[1]-x_q[1]))
    
#                 Eucl_dist = np.sqrt((x_i[0]-x_q[0])**2 + (x_i[1]-x_q[1])**2)
#                 downstream_dist_iq = Eucl_dist*np.cos(abs(theta_iq - wind_dir))
#                 radial_dist_iq =  Eucl_dist*np.sin(abs(theta_iq - wind_dir))
 
#                 r = np.sqrt((radial_dist_iq - np.outer(np.cos(theta_i), r_i) )**2 + (np.outer(np.sin(theta_i), r_i))**2)
#                 d = downstream_dist_iq
                
                
#                 _, Partial_delta_u_Partial_d, Partial_delta_u_Partial_r = wake_model_continuous(d, r, alpha, 2*R0[k])
                
#                 #Partial_d_iq_Partial_x_q = (-(x_i[1]-x_q[1])*np.sin(wind_dir - theta_iq) - (x_i[0]-x_q[0])*np.cos(abs(wind_dir - theta_ik)))/Eucl_dist
#                 Partial_d_iq_Partial_x_q = -np.sin(wind_dir)*np.sign(x_i[1]-x_q[1])
                
#                 #Partial_d_iq_Partial_y_q = ((x_i[0]-x_q[0])*np.sin(wind_dir - theta_iq) - (x_i[1]-x_q[1])*np.cos(abs(wind_dir - theta_ik)))/Eucl_dist
#                 Partial_d_iq_Partial_y_q = -np.cos(wind_dir)*np.sign(x_i[1]-x_q[1])
                
#                 # Matrix-valued expressions
#                 #Partial_r_Partial_x_q = (Eucl_dist*np.sin(abs(wind_dir - theta_ik)) - np.outer(np.cos(theta_i), r_i))*(-Partial_d_Partial_y_i)/r
#                 Partial_r_Partial_x_q = -(radial_dist_iq - np.outer(np.cos(theta_i), r_i))/r*np.cos(wind_dir) * np.sign(x_i[1]-x_q[1])*np.sign(theta_iq-wind_dir)
                
#                 Partial_r_Partial_y_q = (radial_dist_iq - np.outer(np.cos(theta_i), r_i))/r*np.sin(wind_dir) * np.sign(x_i[1]-x_q[1])*np.sign(theta_iq-wind_dir)
                
                
#                 integrand_pts_xder = (Partial_delta_u_Partial_d*Partial_d_iq_Partial_x_q + Partial_delta_u_Partial_r*Partial_r_Partial_x_q)*np.tile(r_i,(n_points_theta,1))
                                
#                 integrand_pts_yder = (Partial_delta_u_Partial_d*Partial_d_iq_Partial_y_q + Partial_delta_u_Partial_r*Partial_r_Partial_y_q)*np.tile(r_i,(n_points_theta,1))
                
                        
                
#                 # Numerical integration Simpson's rule 2D, polar coordinates
                
#                 Partial_delta_u_ik_Partial_x_q[i,q,q] = 1/(np.pi*R0[i]**2)*simps( simps(integrand_pts_xder, theta_i, axis = 0),r_i)
                
#                 Partial_delta_u_ik_Partial_y_q[i,q,q] = 1/(np.pi*R0[i]**2)*simps( simps(integrand_pts_yder, theta_i, axis = 0),r_i)
                
#                 # THESE HAVE BEEN CHANGED NOV 11
#                 Partial_u_i_Partial_x_q[i,q] += Partial_u_i_Partial_delta_u_ik[i,q]*Partial_delta_u_ik_Partial_x_q[i,q,q]
#                 Partial_u_i_Partial_y_q[i,q] += Partial_u_i_Partial_delta_u_ik[i,q]*Partial_delta_u_ik_Partial_y_q[i,q,q]
                    
#     #print("Partial_u_i_Partial_delta_u_ik: ", Partial_u_i_Partial_delta_u_ik)
#     #print("Partial_delta_u_ik_Partial_x_q: ", Partial_delta_u_ik_Partial_x_q)
#     Partial_f_Partial_x_q = np.zeros((N_turb,1))
#     Partial_f_Partial_y_q = np.zeros((N_turb,1))
#     A = np.pi*np.power(R0,2)
#     for q in range(N_turb):
#         # Check cut-in and cut-out speeds below:
#         Partial_f_Partial_x_q[q] = 1.5*rho*C_p*np.sum(A*u_i**2*Partial_u_i_Partial_x_q[:,q])
#         Partial_f_Partial_y_q[q] = 1.5*rho*C_p*np.sum(A*u_i**2*Partial_u_i_Partial_y_q[:,q])
    
    
#     #g = np.zeros((2*N_turb,1))
#     g = np.zeros_like(x_vector)
#     g[::2] = Partial_f_Partial_x_q.flatten()
#     g[1::2] = Partial_f_Partial_y_q.flatten()
    
#     return -P, -g
#     #return delta_u_ij

#@rectangular_area([-100, 100], [-100, 100])
def calc_total_P(x_vector, U, wind_dir, R0, alpha, rho, U_cut_in, U_cut_out, C_p):
    #print("Checking x_vector: ", np.shape(x_vector))
    #    np.atleast_1d(
    N_turb = int(np.shape(x_vector)[0]/2)
    x_all = np.reshape(x_vector, (N_turb, 2))
    D = 2*R0
    #N_turb, temp = np.shape(x_all)
                    
    # Averaged wind and deficit factors
    delta_u_ij = np.zeros((N_turb, N_turb))
    u_i = np.zeros((N_turb,1))
    p_i = np.zeros((N_turb,1))
        
    for i in range(N_turb):
        x_i = x_all[i,:]
        u_i, delta_uij_temp = averaged_wind_speed(x_i, x_all, U, wind_dir, D, R0[i])
        delta_u_ij[i,:] = delta_uij_temp.T
    
        p_i[i] = power_ind_turbine(u_i, U_cut_in, U_cut_out, C_p, rho, R0[i])
        
    return -np.sum(p_i)

# def partial_fun(fun, x, fun_param, dl=0.1):
#     """
#     Calculates the partial derivatives of the function 'fun' at point 'x', of the variables 'x'
#     Args:
#         fun       : is a function taking the arguments (x, *fun_param)
#         x         : the point where the partial derivative is evaluated. (assumin a 1-D ndarray)
#         fun_param : the rest of 'fun''s arguments. (assumin a tuple)
#         dl        : the size of variation used to calculate the partial derivativs. (default = 0.1)
#     Output:
#         dP : list of partial derivative, with the same ordering as the 'x' array. (1-D ndarray)
#     """
#     dP = np.zeros(len(x))
#     dx = np.zeros(x.shape)
#     for i in np.arange(len(x)):
#         dx[i] = dl
#         fun_p, g_tmp = fun(x+dx, *fun_param)
#         fun_n, g_tmp = fun(x-dx, *fun_param)
#         dP[i] = (fun_p - fun_n)/(2*dl)
#         dx[i] = 0
#     return dP

def calc_partial(fun, x, fun_param, dl=0.1):
    """
    Calculates the partial derivatives of the function 'fun' at point 'x', of the variables 'x'
    Args:
        fun       : is a function taking the arguments (x, *fun_param)
        x         : the point where the partial derivative is evaluated. (assumin a 1-D ndarray)
        fun_param : the rest of 'fun''s arguments. (assumin a tuple)
        dl        : the size of variation used to calculate the partial derivativs. (default = 0.1)
    Output:
        dP : list of partial derivative, with the same ordering as the 'x' array. (1-D ndarray)
    """
    dP = np.zeros(len(x))
    dx = np.zeros(x.shape)
    dl_inv = 0.5/dl
    for i in np.arange(len(x)):
        dx[i] = dl
        dP[i] = (fun(x+dx, *fun_param) - fun(x-dx, *fun_param))*dl_inv
        dx[i] = 0
    return dP


def objective_fun_num(x_vector, U, wind_dir, R0, alpha, rho, U_cut_in, U_cut_out, C_p):
    fun_param = U, wind_dir, R0, alpha, rho, U_cut_in, U_cut_out, C_p
    return calc_total_P(x_vector, *fun_param), calc_partial(calc_total_P, x_vector, fun_param)



# Main program

# Induction factor alpha, based on the actuator disc model
# Note that alpha=(U-U_R)/U, so the actual wind speeds upstream and downstream of the disc should be used
alpha = 0.33

U_cut_in = 5
U_cut_out = 20
rho = 1.225 # [kg/m 3]

# Power coefficient
C_p = 4*alpha*(1-alpha)**2



# Coordinates of all wind turbines
x_all = np.array([[-50, 0], [0, 50], [0, -50]])
x_all = np.array([[-80, -80], [-50,-50],[0,0],[25,25],[50,50],[75,75]])
x_all = np.array([[0,0],[-40,40], [40,40], [0,80]])
x_all = np.array([[0,0],[0,30],[0,60],[30,0],[30,30],[30,60],[60,0],[60,30],[60,60]])
x_all = np.array([[0,0],[0,30],[0,60],[60,0],[60,30],[60,60]])
x_all = np.array([[0,0],[0,60],[30,0],[30,60],[60,0],[60,60]])

# Radius and diameter, currently assuming all turbines to be identical
R0 = [10, 10, 20, 20, 10, 10]
R0 = [20, 20, 10, 10]
R0 = [20,20,20,20,20,20] #,20,20,20]
R0 = [20,20]
D = 2*R0

N_turb = len(R0)




#x_i = x_all[0:1]
U = 20

# Set wind direction (radians):
# S: 0; W: pi/2; N: pi; E: 3/2*pi

wind_dir = 0.01 #1.0*np.pi #0.23*np.pi #0.28*np.pi




r_i = np.atleast_2d([0.1])
theta_i = np.atleast_2d([1])

#u_fun = wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D, r_i, theta_i)
#W  = Simpson_2D(0,1,0,1,6,u_fun)

#u_eff, _ = averaged_wind_speed(x_i, x_all, U, wind_dir, D, R0[0])
#print("Averaged wind speed: ", u_eff)

#print()

#exit(1)

#x_vector = np.zeros((2*N_turb,1))
#x_vector[:,0] = np.reshape(x_all, -1)



############################################################################################## ADDED EJ
R0 = [20,20,20,20]
D = 2*R0

x_min = -100*np.ones((8,))
x_max =  100*np.ones((8,))


N_turb = len(R0)
x_all = np.array([[50.,50.0],[-50.,50.0],[-50.,-50.0],[50.,-50.0]])


for wind_dir in np.array([0, 0.25, 0.5, 0.75])*np.pi:#np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75+0.01:
    x_i = x_all[1,:]
    x_vector = np.copy(np.reshape(x_all, -1))

    fun_param = U,wind_dir, R0, alpha, rho, U_cut_in, U_cut_out, C_p

    from wp_optimize import minimize_fun
    x_vector = minimize_fun(calc_total_P, x_vector, fun_param, x_min, x_max, 0.01, 20.0)
    val_min = calc_total_P(x_vector, *fun_param)
    # val_min = 100000000
    # # Test simple minimalization
    # for n in np.arange(100):
    #     dt = 1.0
    #     val, d_val = objective_fun_num(x_vector, *fun_param)
    #     #print(d_val)
    #     norm = np.amax(np.abs(d_val))*np.size(d_val) 
    #     norm = norm if norm>0 else 1.0
    #     #x_vector -= dt*d_val/norm   
    #     #print(val)
    #     #print(x_vector, end="\n\n")
    #     if val < val_min:
    #         x_vector -= dt*d_val/norm
    #         val_min = val
    #     else:
    #         print("val = " + str(val_min) + " ("+str(n) + ")")
    #         break

    print("(final) val = " + str(val_min) + " ("+str(0) + ")")
        
    print_wind_field(np.copy(np.reshape(x_vector, x_all.shape)), U, wind_dir, D, f"basic algorithm {int(180*wind_dir/np.pi)}")

    x_i = x_all[1,:]
    x_vector = np.copy(np.reshape(x_all, -1))
    constr = LinearConstraint(np.identity(2*N_turb), x_vector-50, x_vector+50 )
    obj_fun = functools.partial(objective_fun_num, U=U,wind_dir=wind_dir, R0=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, C_p=C_p)
    res = minimize(obj_fun, x_vector, method='BFGS', jac=True, options={'disp': True}, constraints=None)

    print_wind_field(np.copy(np.reshape(res.x, x_all.shape)), U, wind_dir, D, f"python algorithm {int(180*wind_dir/np.pi)}")

plt.show()


if False:
#================================================================================================== ADDED EJ END
    constr = LinearConstraint(np.identity(2*N_turb), x_vector-50, x_vector+50 )

    wind_dirs = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])*np.pi +0.01

    # Visualize wind field due to wake effects
    N_x = 240
    N_y = 240
    x_grid = np.linspace(-120,120,N_x)
    y_grid = np.linspace(-120,120,N_y)

    #x_grid = np.linspace(-50,100,N_x)
    #y_grid = np.linspace(-50,300,N_y)

    u_eval = np.zeros((N_x, N_y))
    delta_u_eval = np.zeros((N_turb, N_x, N_y))

    u_eval_optim = np.zeros((N_x, N_y))
    delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))


    for ix in range(N_x):
        for iy in range(N_y):
            x_i = (x_grid[ix], y_grid[iy])
            u_eval[ix,iy], temp = wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D, r_i = 0, theta_i = 0)
            delta_u_eval[:,ix,iy] = temp.flatten()

    fig = plt.figure(constrained_layout=True)

    (xv, yv) = np.meshgrid(x_grid, y_grid)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv, yv, u_eval.T, cmap=plt.cm.coolwarm)
    cset = ax.contourf(xv, yv, u_eval.T, zdir='z', offset=25, cmap=cm.coolwarm)
    ax.scatter3D(x_all[:,0], x_all[:,1], np.full((1,N_turb),25),'+k', s=4)


    plt.colorbar(surf, shrink=0.5) # an example

    fig.colorbar(surf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Wind speed, initial conf.')
    ax.azim = -90
    #ax.dist = 10
    ax.elev = 90


    for wind_dir in wind_dirs:

        obj_fun = functools.partial(objective_fun_num, U=U,wind_dir=wind_dir, R0=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, C_p=C_p)

        res = minimize(obj_fun, x_vector, method='BFGS', jac=True, options={'disp': True}, constraints=None)


        print("Initial turbine locations: ", x_vector)
        print("New turbine locations: ", res.x)
        
        for ix in range(N_x):
            for iy in range(N_y):
                x_i = (x_grid[ix], y_grid[iy])

                u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
                delta_u_eval_optim[:,ix,iy] = temp.flatten()
    
        fig = plt.figure(constrained_layout=False)
        (xv, yv) = np.meshgrid(x_grid, y_grid)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xv, yv, u_eval_optim.T, cmap=plt.cm.coolwarm)
        cset = ax.contourf(xv, yv, u_eval_optim.T, zdir='z', offset=25, cmap=cm.coolwarm)
        #ax.scatter3D(res.x[0::2], res.x[1::2], np.full((1,N_turb),25),'k+', s=4)

        #im = ax.imshow(np.arange(200).reshape((20, 10)))
        #plt.colorbar(im,fraction=0.046, pad=0.04)

        #plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        #plt.tight_layout()

        ###ax._axis3don = False

        fig.colorbar(surf)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Wind speed, opt. locations')
        ax.azim = -90
        #ax.dist = 10
        ax.elev = 90

    plt.show()
        
    exit(1)
    #P,g = obj_fun(x_vector)
    #print("Objective function with partial arguments, P: ", P)
    #print("Gradients: ", g)
    # Unconstrained optimization
    res = minimize(obj_fun, x_vector, method='BFGS', jac=True, options={'disp': True}, constraints=None)
    #res = minimize(obj_fun, x_vector, method='Nelder-Mead', options={'disp': True}, constraints=None)
    # Constrained optimization
    #res = minimize(obj_fun, x_vector, method='SLSQP', jac=True, options={'disp': True}, constraints=constr)


    print("New turbine locations: ", res.x)
    #res = minimize(obj_fun, x_all, method='BFGS', jac=True, options={'disp': True})

    P_new,g_new = obj_fun(res.x)
    #P_new = obj_fun(res.x)
    print("Updated objective, P: ", P_new)


    # Visualize wind field due to wake effects
    N_x = 150
    N_y = 600
    x_grid = np.linspace(-100,100,N_x)
    y_grid = np.linspace(-200,500,N_y)

    #x_grid = np.linspace(-50,100,N_x)
    #y_grid = np.linspace(-50,300,N_y)

    u_eval = np.zeros((N_x, N_y))
    delta_u_eval = np.zeros((N_turb, N_x, N_y))

    u_eval_optim = np.zeros((N_x, N_y))
    delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))

    for ix in range(N_x):
        for iy in range(N_y):
            x_i = (x_grid[ix], y_grid[iy])

            u_eval[ix,iy], temp = wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D, r_i = 0, theta_i = 0)
            delta_u_eval[:,ix,iy] = temp.flatten()
            
            u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
            delta_u_eval_optim[:,ix,iy] = temp.flatten()
    
    #fig = plt.figure(constrained_layout=False)
    fig, ax = plt.subplots(constrained_layout=False)
    (xv, yv) = np.meshgrid(x_grid, y_grid)
    plt.imshow(u_eval_optim.T, cmap=cm.coolwarm)
    plt.tight_layout()
    
    
    fig = plt.figure(constrained_layout=True)

    (xv, yv) = np.meshgrid(x_grid, y_grid)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv, yv, u_eval.T, cmap=plt.cm.coolwarm)
    cset = ax.contourf(xv, yv, u_eval.T, zdir='z', offset=25, cmap=cm.coolwarm)
    ax.scatter3D(x_all[:,0], x_all[:,1], np.full((1,N_turb),25),'+k', s=4)

    ###ax._axis3don = False

    plt.colorbar(surf, shrink=0.5) # an example

    fig.colorbar(surf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Wind speed, initial conf.')
    ax.azim = -90
    #ax.dist = 10
    ax.elev = 90

        
    fig = plt.figure(constrained_layout=False)
    (xv, yv) = np.meshgrid(x_grid, y_grid)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xv, yv, u_eval_optim.T, cmap=plt.cm.coolwarm)
    cset = ax.contourf(xv, yv, u_eval_optim.T, zdir='z', offset=25, cmap=cm.coolwarm)
    #ax.scatter3D(res.x[0::2], res.x[1::2], np.full((1,N_turb),25),'k+', s=4)

    #im = ax.imshow(np.arange(200).reshape((20, 10)))
    #plt.colorbar(im,fraction=0.046, pad=0.04)

    #plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    #plt.tight_layout()

    ###ax._axis3don = False

    fig.colorbar(surf)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Wind speed, opt. locations')
    ax.azim = -90
    #ax.dist = 10
    ax.elev = 90

    #cset = ax.contourf(xv, yv, u_eval_optim.T, cmap=cm.coolwarm)



    """
    for j in range(N_turb):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xv, yv, delta_u_eval[j,:,:].T, cmap=plt.cm.coolwarm)

        fig.colorbar(surf)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Wind speed reduction factor')
        

    for j in range(N_turb):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xv, yv, delta_u_eval_optim[j,:,:].T, cmap=plt.cm.coolwarm)

        fig.colorbar(surf)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Wind speed reduction factor, optimized')
    """

    plt.show()


    # Compute power production for all turbines:

    power = np.zeros((N_turb,1))
    u_at_turbines = np.zeros((N_turb,1))
    for j in range(N_turb):
        # Averaged speed over the disc centered at site j
        u_at_turbines[j], temp = averaged_wind_speed(x_all[j,:], x_all, U, wind_dir, D, R0[j])
        power[j] = power_ind_turbine(u_at_turbines[j], U_cut_in, U_cut_out, C_p, rho, R0[j])
        print("Power of turbine ", j, ": ", power[j])
        


