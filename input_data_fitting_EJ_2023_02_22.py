import pyvinecopulib as pv
import numpy as np
from scipy.integrate import simps
from matplotlib import pyplot as plt
from scipy.stats import norm, uniform
import scipy.stats as stats
import itertools
import functools
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from matplotlib import cm
import sys
import pylab
import datetime
import windfarmmodel as wfm

from scipy.special import rel_entr
from pylab import *

plt.rcParams['text.usetex'] = True

def repeat_product(x, y):
    #return np.transpose([np.tile(x, len(y)),
    #                        np.repeat(y, len(x))])
    return np.transpose([np.repeat(x, len(y)),
                            np.tile(y, len(x))])                        

def test_obj(w_speed, w_dir, design_var) -> np.ndarray:
    # Test function that 'produces' more when faced parallel w. wind dir
     
    angle = design_var
    P = w_speed*np.cos((angle-w_dir)*np.pi/180)


    return P
    
    
def robust_design_func(design_var, pts_phys, wts):
    
    P_evals = test_obj(pts_phys[0,:],pts_phys[1,:], design_var)
    mu = np.sum(P_evals*wts)
    sigma = np.sqrt(np.sum(P_evals**2*wts)-mu**2)
    
    print("mu: ", mu, "sigma: ", sigma, "-(mu-sigma): ", -(mu-sigma))
    return -(mu-sigma)
    


"""
R : float (default = 63.0)
    Actuator disk radius
Uin : float (default = 3.0)
    Cut in wind speed. Minimum speed for power production
Uout : float (default = 12.0)
    Cut out speed. Maxiumum power production
rho : float (default = 1.0)
    Air density
kappa :
    Wake expansion rate
alpha : float (default = 1/3)  
    Induction factor
"""

def objective_fun_num_robust_design(x_vector, pts, wts, R0_loc, alpha, rho, U_cut_in, U_cut_out, U_stop, C_p):
    kappa = 0.05
    sigma_weight = 0.1
    Nq = len(wts)
    N_turb = int(len(x_vector)/2.)
    dPdx_evals = np.zeros((Nq,2*N_turb))
    #P_evals = np.zeros((Nq,1))
    P_evals = np.zeros(Nq)

    for q in range(Nq):
        U = pts[0,q]
        wind_dir = pts[1,q]
        fun_param = wind_dir, R0_loc[0], U_cut_in, U_cut_out, rho, kappa , alpha, U
        P_evals[q] = calc_total_P(x_vector, *fun_param)
        #temp = calc_partial(calc_total_P, x_vector, fun_param)
        dPdx_evals[q,:] =  calc_partial(calc_total_P, x_vector, fun_param)


        #print("dPdx_evals[q,:]: ", dPdx_evals[q,:])
    mu = np.sum(P_evals*wts)
    wts = wts.reshape(1, -1)
    grad_mu = wts.dot(dPdx_evals)
    #print("grad_mu: ", grad_mu)
    sigma = np.sqrt(np.sum(P_evals**2*wts)-mu**2)

    
    if sigma == 0:

        print("Turbine positions: ", x_vector)    
        print("mu: ", mu, ", sigma: ", sigma, ", (mu+sigma_weight*sigma): ", (mu+sigma_weight*sigma))
        return mu, grad_mu

    else:
        grad_sigma_sq = wts.dot(2*np.tile(np.c_[P_evals],(1,2*N_turb))*dPdx_evals) - 2*mu*grad_mu
        grad_sigma = grad_sigma_sq/(2.*sigma)
    
        #print("grad_sigma: ", grad_sigma )
        print("Turbine positions: ", x_vector)    
        print("mu: ", mu, ", sigma: ", sigma, ", (mu+sigma_weight*sigma): ", (mu+sigma_weight*sigma))
        #return mu, grad_mu #(mu-sigma)
        return mu+sigma_weight*sigma, grad_mu+sigma_weight*grad_sigma
    
    
def power_ind_turbine(U_loc, U_cut_in_loc, U_cut_out_loc, U_stop_loc, C_p_loc, rho_loc, R0_loc):
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
    A = np.pi*R0_loc**2
    if U_loc < U_cut_in_loc or U_loc>U_stop_loc:
        Power = 0
    elif U_loc > U_cut_out_loc:
        Power = 0.5*rho_loc*A*C_p_loc*U_cut_out_loc**3
    else:
        Power = 0.5*rho_loc*A*C_p_loc*U_loc**3
    """
    print("U =", U)
    print("R0_loc =", R0_loc)
    print("rho =", rho)
    print("C_p =", C_p)
    print("U_cut_out =", U_cut_out)
    print("Power P=", Power)    
    """
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
        if downstream_dist_ij  <= 0:
            delta_u_i[j] = 0
        
    delta_u = np.sqrt(np.sum(delta_u_i**2, axis=0))

    
    u_i = U*(1-delta_u)
    
    #if downstream_dist_ij  >= 0:
    #    u_ij = U*(1-delta_u)
    #else:
    #    u_ij = U
    

    return u_i, delta_u_i
        
    
def wake_model_continuous(d, r, alpha, D_loc, averaged=False):
    #
    # Linear wake expansion
    kappa = 0.05
    R0_loc = 0.5*D_loc
    R_loc = R0_loc + kappa*d
    #deficit factor Eq. (9)
    delta_u = 2*alpha*(R0_loc/R_loc)**2*np.exp(-(r/R_loc)**2)
    
    #Partial_delta_u_Partial_d = -4*alpha*R0**2*kappa*np.exp(-r**2/R**2)*(R**(-3)+r/R)
    Partial_delta_u_Partial_d = 4*alpha*R0_loc**2*kappa/R_loc**3*np.exp(-r**2/R_loc**2)*(r**2/R_loc**2-1)

    Partial_delta_u_Partial_r = -2*r/R_loc**2*delta_u
        
    return delta_u, Partial_delta_u_Partial_d, Partial_delta_u_Partial_r
    

def averaged_wind_speed(x_i, x_all, U, wind_dir, D_loc, R0_loc):
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
    r_min, r_max, n_points_r = (1e-6, R0_loc, 20)
    theta_i = np.linspace(theta_min, theta_max, n_points_theta)
    r_i = np.linspace(r_min, r_max, n_points_r)

    #print("r_i =", r_i)

    # Get wake-reduced wind speed point-wise
    u_evals, delta_u_evals = wind_speed_due_to_wake(x_i, x_all, U, wind_dir, D_loc, r_i, theta_i)
    integrand_pts_u = u_evals*np.tile(r_i,(n_points_theta,1))
    integrand_pts_delta_u = delta_u_evals*np.tile(r_i,(n_points_theta,1))
   

    # Numerical integration Simpson's rule 2D, polar coordinates
    u_averaged = 1/(np.pi*R0_loc**2)*simps(simps(integrand_pts_u, theta_i, axis = 0),r_i)

    delta_u_averaged = 1/(np.pi*R0_loc**2)*simps(simps(integrand_pts_delta_u, theta_i, axis = 1),r_i, axis=1)
    
    """
    print("averaged_wind_speed, D = ", D_loc)
    print("averaged_wind_speed, R0 = ", R0_loc)
    print("averaged_wind_speed, U = ", U)
    print("averaged_wind_speed, u_averaged = ", u_averaged)
    """

    return u_averaged, delta_u_averaged


def calc_total_P(x_vector, wind_dir, R, Uin, Uout, rho, kappa , alpha, U):
    N_turb = len(x_vector)//2
    wt_pos = np.zeros((3, N_turb))
    wt_pos[0, :] = x_vector[::2]
    wt_pos[1, :] = x_vector[1::2]

    ang = 0.5*np.pi - wind_dir*np.pi/180 
    vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

    return -wfm.Windfarm(wt_pos, vec_dir, R, Uin, Uout, rho, kappa, alpha).power(U)


def calc_total_P_old(x_vector, U, wind_dir, R0_loc, alpha, rho, U_cut_in, U_cut_out, U_stop, C_p):
    #print("Checking x_vector: ", np.shape(x_vector))
    #    np.atleast_1d(
    N_turb = int(np.shape(x_vector)[0]/2.)
    x_all = np.reshape(x_vector, (N_turb, 2))
    D_loc = 2*R0_loc
    #N_turb, temp = np.shape(x_all)

    # Averaged wind and deficit factors
    delta_u_ij = np.zeros((N_turb, N_turb))
    u_i = np.zeros((N_turb,1))
    p_i = np.zeros((N_turb,1))
        


    for i in range(N_turb):
        x_i = x_all[i,:]
        u_i, delta_uij_temp = averaged_wind_speed(x_i, x_all, U, wind_dir, D_loc, R0_loc[i])

       

        delta_u_ij[i,:] = delta_uij_temp.T
    
        p_i[i] = power_ind_turbine(u_i, U_cut_in, U_cut_out, U_stop, C_p, rho, R0_loc[i])
        
    return -np.sum(p_i)

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

    #NBNBNBNBNBNBNBNBNB bare for testing
    dP= dP+2e-1*(np.random.random_sample(dP.shape) - 0.5)   
    #NBNBNBNBNBNBNBNBNB slutt
    return dP
    
def objective_fun_num(x_vector, U, wind_dir, R0_loc, alpha, rho, U_cut_in, U_cut_out, U_stop, C_p):
    kappa = 0.05
    fun_param = wind_dir, R0_loc[0], U_cut_in, U_cut_out, rho, kappa , alpha, U
    return calc_total_P(x_vector, *fun_param), calc_partial(calc_total_P, x_vector, fun_param)

def cons_c(x_vector, c_J_ind):
    N_turb = int(len(x_vector)/2.)
    x = x_vector.reshape((N_turb, 2))
    
    num_con, _ = np.shape(c_J_ind)
    cons = np.zeros(num_con)
    for row_ind in range(num_con):
        q = int(c_J_ind[row_ind,0])
        m = int(c_J_ind[row_ind,1])
        
        cons[row_ind] = (x_vector[2*q]-x_vector[2*m])**2 + (x_vector[2*q+1]-x_vector[2*m+1])**2
        #cons[row_ind] = (x[q,0]-x[m,0])**2 + (x[q,1]-x[m,1])**2
    return cons
 

def cons_J(x_vector, c_J_ind):
    N_turb = int(len(x_vector)/2.)
    x = x_vector.reshape((N_turb, 2))
    
    num_con, _ = np.shape(c_J_ind)
    Jac = np.zeros((num_con,N_turb*2))
    
    for row_ind in range(num_con):
        q = int(c_J_ind[row_ind,0])
        m = int(c_J_ind[row_ind,1])
        
        Jac[row_ind,2*q] = 2*(x[q,0]-x[m,0])
        Jac[row_ind,2*q+1] = 2*(x[q,1]-x[m,1])
        Jac[row_ind,2*m] = -2*(x[q,0]-x[m,0])
        Jac[row_ind,2*m+1] = -2*(x[q,1]-x[m,1])
        
    return Jac
    



#pathName = '/Users/pepe/Documents/MATLAB/wind_Matlab/'
#pathName = "C:/Users/jhel/" #'Documents/MATLAB/wind_Matlab/'
#pathName = '/home/AD.NORCERESEARCH.NO/olau/Documents/projects/DynPosWind/opt_farm/data/'
pathName = "/home/AD.NORCERESEARCH.NO/esje/Programs/GitHub/windpark-optimization/data/"


theta_data  = np.loadtxt(pathName + 'Dir_100m_2016_2017_2018.txt')
u_data  = np.loadtxt(pathName + 'Sp_100m_2016_2017_2018.txt')

#theta_data = np.array([0.])
#u_data = np.array([25.])

theta_data[theta_data<0] = theta_data[theta_data<0] + 360
theta_data[theta_data>360] = theta_data[theta_data>360] - 360

print("Theta, min and max: ",np.amin(theta_data), np.amax(theta_data))


if False:

    u_theta_data = np.array([u_data, theta_data])
    print("np.shape(u_theta_data): ", np.shape(u_theta_data))

    u = pv.to_pseudo_obs(u_theta_data.T)
    cop = pv.Vinecop(data=u)
    print(cop)
    pv.Vinecop.to_json(cop, "copula_tree_100m_2016_2017_2018.txt")

#pathName = '/Users/pepe/Documents/Wind/Pywake/'
#pathName = "C:/Users/jhel/" 
#pathName = '/home/AD.NORCERESEARCH.NO/olau/Documents/projects/DynPosWind/opt_farm/data/'
    
data_samples = np.array([u_data, theta_data])
    
# Instantiate copula object from precomputed model saved to file
copula = pv.Vinecop(filename = pathName + 'copula_tree_100m_2016_2017_2018.txt', check=True)
print(copula)

num_samples = 10000 #Number of model evaluations
num_samples = 26304
num_dim = 2

# Some physical/technical parameters must be introduced already here
# Induction factor alpha, based on the actuator disc model
alpha = 0.3333333
# Note that alpha=(U-U_R)/U, so the actual wind speeds upstream and downstream of the disc should be used

U_cut_in = 3 #5
U_cut_out = 10.59 #10 #25 #20
U_stop = 25.
rho = 1.225#1.225 # [kg/m 3]

# Power coefficient
#C_p = 4*alpha*(1-alpha)**2
C_p= 0.489

#Olav 
opt_tolerance_SLSQP = 5e-01
maxiter_SLSQP = 200
now = datetime.datetime.now()
np.random.seed(0)
rotorRadius = 120. #50.

wake_model_param = np.array([C_p, rho, U_cut_in, U_cut_out])


# Independent uniforms
u_samples_ind = np.random.uniform(low=0, high=1, size=(num_samples, num_dim))

# Sample copula via inverse Rosenblatt
cop_samp_from_ind = copula.inverse_rosenblatt(u_samples_ind)


# Transform back simulations to the original scale
copula_samples_from_unif = np.asarray([np.quantile(data_samples[i,:], cop_samp_from_ind[:, i]) for i in range(0, num_dim)])


# Construct quadrature rule for expectation operators
# Divide parameter domain w.r.t. U_cut_in and U_cut_off
# Introduce point probability masses, and local GQ


p_0 = np.sum((u_data < U_cut_in) | (u_data >= U_stop))/num_samples
ind_var_s = (u_data < U_cut_out) & (u_data >= U_cut_in)
ind_con_s = (u_data >= U_cut_out) & (u_data < U_stop)

#p_0 = np.sum(u_data < U_cut_in)/num_samples
#ind_var_s = (u_data < U_cut_out) & (u_data >= U_cut_in)
#ind_con_s = u_data >= U_cut_out

p_1 = np.sum(ind_var_s)/num_samples
p_2 = np.sum(ind_con_s)/num_samples

#print("Probabilities: ", p_0, p_1, p_2, np.sum(p_0+p_1+p_2))

# Set number of quadrature points for the regions 1 ([U_cut_in, U_cut_out]) and 2 ([U_cut_out, infty])
# in speed (sp) and direction (dir)
Nq_sp = np.array([5,3])
Nq_dir = np.array([10,10])

[pts_sp_1, wts_sp_1] = np.polynomial.legendre.leggauss(Nq_sp[0])
[pts_dir_1, wts_dir_1] = np.polynomial.legendre.leggauss(Nq_dir[0])

[pts_sp_2, wts_sp_2] = np.polynomial.legendre.leggauss(Nq_sp[1])
[pts_dir_2, wts_dir_2] = np.polynomial.legendre.leggauss(Nq_dir[1])

# Rescale to unit interval, and weights summing to 1
pts_sp_1 = (pts_sp_1+1)/2.
wts_sp_1 = 0.5*wts_sp_1
pts_dir_1 = (pts_dir_1+1)/2.
wts_dir_1 = 0.5*wts_dir_1

pts_sp_2 = (pts_sp_2+1)/2.
wts_sp_2 = 0.5*wts_sp_2
pts_dir_2 = (pts_dir_2+1)/2.
wts_dir_2 = 0.5*wts_dir_2

#print("Quadrature tests 1D: ", sum(wts), sum(pts*wts), sum((pts**2)*wts), sum(pts**3*wts))
# Quadrature for U in [U_cut_in, U_cut_out]
pts_2D_1 = repeat_product(p_0+p_1*pts_sp_1, pts_dir_1).T
wts_2D_1 = np.kron(wts_sp_1, wts_dir_1)

# Quadrature for U > U_cut_out
pts_2D_2 = repeat_product(p_0+p_1+p_2*pts_sp_2, pts_dir_2).T
wts_2D_2 = np.kron(wts_sp_2, wts_dir_2)

# Form compound quadrature rule
pts_2D_comp = np.concatenate((np.zeros((2, 1)), pts_2D_1, pts_2D_2), axis=1)
wts_2D_comp = np.concatenate(([p_0], p_1*wts_2D_1,p_2*wts_2D_2), axis=0)

# May want to compare to standard GQ, for now use only compound rule
pts_2D = pts_2D_comp
wts_2D = wts_2D_comp



# Evaluate copula via inverse Rosenblatt
cop_evals = copula.inverse_rosenblatt(pts_2D.T)

# Transform back evaluations to the original scale
cop_evals_physical = np.asarray([np.quantile(data_samples[i,:], cop_evals[:, i]) for i in range(0, num_dim)])

print("Size cop_evals_physical: ", np.shape(cop_evals_physical))

print("cop_evals_physical: ", cop_evals_physical)


print("GQ copula: ", np.sum(np.prod(cop_evals_physical, axis=0)*wts_2D))

print("MC copula: ", np.mean(np.prod(copula_samples_from_unif, axis=0)))

print("MC data: ", np.mean(np.prod(data_samples, axis=0)))




print("-------------------------------------------------------")
print('start time: ', now)
print("-------------------------------------------------------")

# JAN 2023, try opt under unc.


# Coordinates of all wind turbines

#x_all = np.array([[0.,0.]])
x_all = np.array([[0, 0],[240, 240]])

#x_all = np.array([[0,0],[100,100],[100,-100],[-100,100],[-100,-100]])
#x_all = np.array([[0,0],[100,100],[100,-100],[-100,100],[-100,-100],[0,200],[0,-200],[200,0],[-200,0]])
#x_all = np.array([[0,0],[20,20],[20,-20],[-20,20],[-20,-20],[0,40],[0,-40],[40,0],[-40,0],[40,40],[40,-40],[-40,40],[-40,-40]])
#x_all = np.array([[0,0],[20,20],[20,-20],[-20,20],[-20,-20],[0,40],[0,-40],[40,0],[-40,0],[40,40],[40,-40],[-40,40],[-40,-40], [60,20],[60,-20],[-60,20],[-60,-20]])

#x_all = np.array([[-4500, 3000], [-2250, 3000], [0,3000], [2250, 3000], [4500,3000],[-4500, 1000], [-2250, 1000], [0,1000], [2250, 1000], [4500,1000], [-4500, -1000], [-2250, -1000], [0, -1000], [2250, -1000], [4500, -1000], [-4500, -3000], [-2250, -3000], [0,-3000], [2250, -3000], [4500,-3000]])
#x_all = np.array([[-2000, 2500],[-1000,2500],[0,2500],[1000, 2500],[2000, 2500]])
print("x_all.shape", x_all.shape)

print("x_all", x_all)

x_vector = x_all.flatten()
print("x_vector", x_vector, "type ", x_vector.dtype)


# Radius and diameter, currently assuming all turbines to be identical
#R0 = [20,20,20,20,20,20,20,20,20]
#R0 = [10,10,10,10,10,10,10,10,10]

R0 = np.ones(len(x_all))*rotorRadius

print("length of x_all:", len(x_all))

print("R0:", R0)

print("max power 1 turbine, P=", 0.5*rho*np.pi*R0[0]**2*C_p*U_cut_out**3/1e6,"MW")



R_Constraint = (2*R0[0])**2 

D = 2*R0

print("D:", D)
print("R0:", R0)


N_turb = len(R0)
print("Number of turbines:", N_turb)

#U = 20.0
# Set wind direction (radians):
# S: 0; W: pi/2; N: pi; E: 3/2*pi

#wind_dir = 0.25*np.pi #1.0*np.pi #0.23*np.pi #0.28*np.pi

#lin_constr = LinearConstraint(np.identity(2*N_turb), x_vector-50, x_vector+50 )

xmin = -5000 #-41.6667*R0[0]#-375 #-250
xmax = 5000 #41.6667*R0[0]#-375 #250
ymin = -3000 #-25*R0[0]#-250
ymax = 3000#25*R0[0]#250

confinement_rectangle = np.array([xmin, ymin, xmax, ymax]) 

lin_constr = LinearConstraint(np.identity(2*N_turb), np.tile((xmin,ymin),N_turb), np.tile((xmax,ymax),N_turb))
 
N_c_J = np.arange(0,N_turb)
c_J_ind = np.zeros((np.sum(N_c_J),2))
tot_ind = 0
for first_ind in N_c_J:
    sec_ind_range = np.arange(first_ind+1,N_turb)
    for sec_ind in sec_ind_range:
        c_J_ind[tot_ind,:] = [first_ind, sec_ind]
        tot_ind += 1

print("c_J_ind: ", c_J_ind)

cons_c = functools.partial(cons_c, c_J_ind=c_J_ind)
cons_J = functools.partial(cons_J, c_J_ind=c_J_ind)

nonlinear_constraint = NonlinearConstraint(cons_c, R_Constraint, np.inf, jac=cons_J)
#exit(1)

# Visualize wind field due to wake effects
N_x = 240
N_y = 240
#x_grid = np.linspace(-N_x/2,N_x/2,N_x)
#y_grid = np.linspace(-N_y/2,N_y/2,N_y)
x_grid = np.linspace(-50*R0[0], 50*R0[0], N_x)
y_grid = np.linspace(-50*R0[0], 50*R0[0], N_y)

U = 2.0 #np.mean(cop_evals_physical[0,:])
wind_dir = np.mean(cop_evals_physical[1,:])*np.pi/180. + np.pi


u_eval = np.zeros((N_x, N_y))
delta_u_eval = np.zeros((N_turb, N_x, N_y))

u_eval_optim = np.zeros((N_x, N_y))
delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))



print("-----------")
print("Initial turbine Production: ")
print("-----------")
objective_fun_num_robust_design(x_vector, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p) # OLAV 
print("-----------")

obj_fun = functools.partial(objective_fun_num_robust_design, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p)

#res = minimize(obj_fun, x_vector, method='SLSQP', jac=True, options={'disp': True}, constraints=[lin_constr, nonlinear_constraint]) #constraints=lin_constr) 
res = minimize(obj_fun, x_vector, method='SLSQP', jac=True, options={'disp': True, 'ftol': opt_tolerance_SLSQP, 'maxiter': maxiter_SLSQP}, constraints=[lin_constr, nonlinear_constraint]) # default: 'ftol': 1e-06

fileName = pathName+'arrays1/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.npz'
np.savez(fileName, init_pos=x_vector, end_pos=res.x, rotor_rad=R0, wake_param=wake_model_param, confine_rectangle = confinement_rectangle)

x_opt = np.reshape(res.x, (N_turb,2))
print("Initial turbine locations: ", x_vector)
print("Objective fun value, init: ", obj_fun(x_vector))

print("Robust design, New turbine locations: ", res.x)

print("Objective fun value, opt: ", res.fun)
for ix in range(N_x):
    for iy in range(N_y):
        x_i = (x_grid[ix], y_grid[iy])

        u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
        delta_u_eval_optim[:,ix,iy] = temp.flatten()
    

(xv, yv) = np.meshgrid(x_grid, y_grid)

fig = plt.figure(constrained_layout=True)
fig.set_size_inches(4.0, 4.0)
#cmap2 = plt.get_cmap("jet")
im1= plt.imshow(u_eval_optim.T, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
# interpolation='bicubic'
plt.scatter(x_all[:,0], x_all[:,1], s=50, c='blue', marker='+')
fig.colorbar(im1)
plt.xlabel('$X$ [m]')
plt.ylabel('$Y$ [m]', rotation=0)
plt.title('Wind speed, opt. locations')

###
###
fig1 = plt.figure(constrained_layout=True) #JOH
fig1.set_size_inches(4.2, 3.5) #JOH
im1= plt.imshow(u_eval_optim.T, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
#cset = plt.contourf(xv, yv, u_eval_optim.T, cmap=cm.coolwarm)
#plt.scatter(x_all[:,0], x_all[:,1], s=50, c='black', marker='+')
#plt.scatter(x_opt[:,0], x_opt[:,1], s=30, c='black', marker='o')
plt.plot([xmin, xmin], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmax, xmax], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymin, ymin], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymax, ymax], linestyle='dashed', color='black')

# plt.circle( (x_opt[:,0], x_opt[:,1] ), 10 , fill = False, linestyle='--', color='black' ) #JOH

# JOH comment:
for x, y in zip(x_opt[:,0], x_opt[:,1]):
    circle = plt.Circle((x,y), radius=R0[0], fill=False, linestyle='--', color='black')
    fig=plt.gcf()
    ax=fig.gca()
    ax.add_patch(circle)
    # circle = pylab.Circle((x,y), radius=10, fill=False, linestyle='--', color='black')
    # axes=pylab.axes()
    # axes.add_patch(circle)
                                      
#fig.colorbar(cset)
ax.set_aspect('equal') #JOH
fig1.colorbar(im1)
plt.xlabel('$X\ [\mathrm{m}]$')
plt.ylabel('$Y\ [\mathrm{m}]$', rotation=90)
plt.title('Robust design') #('Wind speed, opt. locations')


#figname = '/Users/pepe/Documents/Wind/figures/Robust_design_Nturb_' + str(N_turb) + '.png'
#figname = 'C:/Users/jhel/Wind_figs/Robust_design_Nturb_' + str(N_turb) + '.png'
figname = pathName+'figures3/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.png'
plt.savefig(figname)

print("-------------------------------------------------------")
print('start time: ', now)
now = datetime.datetime.now()
print('end Robust design time: ', now)
print("-------------------------------------------------------")

###
###
print("-------------------------------------------------------")
print("---------------HER BEGYNNER MEAN-OPT:-------------------")

obj_fun = functools.partial(objective_fun_num, U=U,wind_dir=wind_dir, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p)


res = minimize(obj_fun, x_vector, method='SLSQP', jac=True, options={'disp': True, 'ftol': opt_tolerance_SLSQP, 'maxiter': maxiter_SLSQP}, constraints=[lin_constr, nonlinear_constraint]) #constraints=lin_constr)
x_opt = np.reshape(res.x, (N_turb,2))

print("-------------------------------------------------------")
print("---------------HER KOMMER TALLENE VI TRENGER:-------------------")
objective_fun_num_robust_design(res.x, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p) # OLAV 
#obj_fun = functools.partial(objective_fun_num_robust_design, pts=cop_evals_physical, wts=wts_2D, R0=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, C_p=C_p)

print("Initial turbine locations: ", x_vector)
print("New turbine locations, mean dir and speed: ", res.x)
print("Objective fun value, init: ", obj_fun(x_vector))
print("Objective fun value, opt: ", res.fun)
for ix in range(N_x):
    for iy in range(N_y):
        x_i = (x_grid[ix], y_grid[iy])

        u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
        delta_u_eval_optim[:,ix,iy] = temp.flatten()


###
now = datetime.datetime.now()
print("-------------------------------------------------------")
print('end mean-wind opt. time: ', now)
print("-------------------------------------------------------")

#(xv, yv) = np.meshgrid(x_grid, y_grid)
fig2 = plt.figure(constrained_layout=True)
fig2.set_size_inches(4.2, 3.5)
im1= plt.imshow(u_eval_optim.T, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
#cset = plt.contourf(xv, yv, u_eval_optim.T, cmap=cm.coolwarm) #zdir='z', offset=25, cmap=cm.coolwarm)
plt.scatter(x_all[:,0], x_all[:,1], s=50, c='black', marker='+')
plt.scatter(x_opt[:,0], x_opt[:,1], s=30, c='black', marker='o')
plt.plot([xmin, xmin], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmax, xmax], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymin, ymin], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymax, ymax], linestyle='dashed', color='black')

# JOH comment:
# for x, y in zip(x_opt[:,0], x_opt[:,1]):
#     circle = pylab.Circle((x,y), radius=10, fill=False, linestyle='--', color='black')
#     axes=pylab.axes()
#     axes.add_patch(circle)
# JOH
# 
#  



for x, y in zip(x_opt[:,0], x_opt[:,1]):
    circle = plt.Circle((x,y), radius=R0[0], fill=False, linestyle='--', color='black')
    fig2=plt.gcf()
    ax=fig2.gca()
    ax.add_patch(circle)


#fig.colorbar(cset)
fig2.colorbar(im1)
ax.set_aspect('equal') #JOH
plt.xlabel('$X\ [\mathrm{m}]$')
plt.ylabel('$Y\ [\mathrm{m}]$', rotation=90)
plt.title('Mean wind conditions') #('Wind speed, opt. locations')
#figname = '/Users/pepe/Documents/Wind/figures/Mean_cond_Nturb_' + str(N_turb) + '.png'
#figname = 'C:/Users/jhel/Wind_figs/Mean_cond_Nturb_' + str(N_turb) + '.png'
figname = pathName+'figures3/Mean_cond_Nturb_' + str(N_turb) + '_1.png'

plt.savefig(figname)

###




plt.show()
      
exit(1)

