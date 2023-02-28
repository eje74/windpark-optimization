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
        fun_param = wind_dir, R0_loc[0], U_cut_in, U_cut_out, rho, kappa , C_p, U
        P_evals[q] = calc_total_P(x_vector, *fun_param)
        #temp = calc_partial(calc_total_P, x_vector, fun_param)
        dPdx_evals[q,:] =  calc_partial(calc_total_P, x_vector, fun_param)


        #print("dPdx_evals[q,:]: ", dPdx_evals[q,:])
    mu = np.sum(P_evals*wts)
    wts = wts.reshape(1, -1)
    grad_mu = wts.dot(dPdx_evals)
    #print("grad_mu: ", grad_mu)
    sigma = np.sqrt(np.sum(P_evals**2*wts)-mu**2)

    power_moments = np.array([mu, sigma])
    
    if sigma == 0:

        print("Turbine positions: ", x_vector)    
        print("mu: ", mu, ", sigma: ", sigma, ", (mu+sigma_weight*sigma): ", (mu+sigma_weight*sigma))
        return mu, grad_mu, power_moments

    else:
        grad_sigma_sq = wts.dot(2*np.tile(np.c_[P_evals],(1,2*N_turb))*dPdx_evals) - 2*mu*grad_mu
        grad_sigma = grad_sigma_sq/(2.*sigma)
    
        #print("grad_sigma: ", grad_sigma )
        print("Turbine positions: ", x_vector)    
        print("mu: ", mu, ", sigma: ", sigma, ", (mu+sigma_weight*sigma): ", (mu+sigma_weight*sigma))
        #return mu, grad_mu #(mu-sigma)
        return mu+sigma_weight*sigma, grad_mu+sigma_weight*grad_sigma , power_moments
    


def calc_total_P(x_vector, wind_dir, R, Uin, Uout, rho, kappa , C_p, U):
    N_turb = len(x_vector)//2
    wt_pos = np.zeros((3, N_turb))
    wt_pos[0, :] = x_vector[::2]
    wt_pos[1, :] = x_vector[1::2]

    ang = 0.5*np.pi - wind_dir #*np.pi/180 
    vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

    return -wfm.Windfarm(wt_pos, vec_dir, R, Uin, Uout, rho, kappa, C_p).power(U)



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
    #fun_param = wind_dir, R0_loc[0], U_cut_in, U_cut_out, rho, kappa , alpha, U
    fun_param = wind_dir, R0_loc[0], U_cut_in, U_cut_out, rho, kappa , C_p, U
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
    
def copulaGeneration():
    theta_data  = np.loadtxt(pathName + 'Dir_100m_2016_2017_2018.txt')
    u_data  = np.loadtxt(pathName + 'Sp_100m_2016_2017_2018.txt')

    theta_data[theta_data<0] = theta_data[theta_data<0] + 360
    theta_data[theta_data>360] = theta_data[theta_data>360] - 360


    if False:

        u_theta_data = np.array([u_data, theta_data])
        print("np.shape(u_theta_data): ", np.shape(u_theta_data))

        u = pv.to_pseudo_obs(u_theta_data.T)
        cop = pv.Vinecop(data=u)
        print(cop)
        pv.Vinecop.to_json(cop, "copula_tree_100m_2016_2017_2018.txt")


    data_samples_loc = np.array([u_data, theta_data])

    # Instantiate copula object from precomputed model saved to file
    copula_loc = pv.Vinecop(filename = pathName + 'copula_tree_100m_2016_2017_2018.txt', check=True)

    return copula_loc, data_samples_loc


def copulaDataGeneration(copula_loc, data_samples_loc, Nq_sp_loc, Nq_dir_loc, U_cut_in_loc, U_cut_out_loc, U_stop_loc):

  

    print('shape:',data_samples_loc.shape)
    u_data = data_samples_loc[0,:]
    #theta_data = data_samples_loc[1,:]

    #u_data = data_samples

    # Instantiate copula object from precomputed model saved to file
    copula = pv.Vinecop(filename = pathName + 'copula_tree_100m_2016_2017_2018.txt', check=True)
    #print(copula)

    num_samples = data_samples_loc.shape[1] #Number of model evaluations
    num_dim = data_samples_loc.shape[0]

    # Independent uniforms
    u_samples_ind = np.random.uniform(low=0, high=1, size=(num_samples, num_dim))

    # Sample copula via inverse Rosenblatt
    cop_samp_from_ind = copula_loc.inverse_rosenblatt(u_samples_ind)


    # Transform back simulations to the original scale
    copula_samples_from_unif = np.asarray([np.quantile(data_samples_loc[i,:], cop_samp_from_ind[:, i]) for i in range(0, num_dim)])


    # Construct quadrature rule for expectation operators
    # Divide parameter domain w.r.t. U_cut_in and U_cut_off
    # Introduce point probability masses, and local GQ
    
    #p_0 = np.sum(u_data < U_cut_in)/num_samples
    #ind_var_s = (u_data < U_cut_out) & (u_data >= U_cut_in)
    #ind_con_s = u_data >= U_cut_out

    p_0 = np.sum((u_data < U_cut_in_loc) | (u_data >= U_stop_loc))/num_samples
    ind_var_s = (u_data < U_cut_out_loc) & (u_data >= U_cut_in_loc)
    ind_con_s = (u_data >= U_cut_out_loc) & (u_data < U_stop_loc)

    p_1 = np.sum(ind_var_s)/num_samples
    p_2 = np.sum(ind_con_s)/num_samples

    #print("Probabilities: ", p_0, p_1, p_2, np.sum(p_0+p_1+p_2))

    # Set number of quadrature points for the regions 1 ([U_cut_in, U_cut_out]) and 2 ([U_cut_out, infty])
    # in speed (sp) and direction (dir)
    

    [pts_sp_1, wts_sp_1] = np.polynomial.legendre.leggauss(Nq_sp_loc[0])
    [pts_dir_1, wts_dir_1] = np.polynomial.legendre.leggauss(Nq_dir_loc[0])

    [pts_sp_2, wts_sp_2] = np.polynomial.legendre.leggauss(Nq_sp_loc[1])
    [pts_dir_2, wts_dir_2] = np.polynomial.legendre.leggauss(Nq_dir_loc[1])

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
    cop_evals = copula_loc.inverse_rosenblatt(pts_2D.T)

    # Transform back evaluations to the original scale
    cop_evals_physical = np.asarray([np.quantile(data_samples_loc[i,:], cop_evals[:, i]) for i in range(0, num_dim)])

    cop_evals_physical[1,:] = cop_evals_physical[1,:]*np.pi/180. + np.pi

    print('windDataGeneration() done')

    return cop_evals_physical, wts_2D
    #Returns wind directions in radians

    ######################################################################################################

pathName = '/home/AD.NORCERESEARCH.NO/olau/Documents/projects/DynPosWind/opt_farm/data/'
#pathName = '/Users/pepe/Documents/MATLAB/wind_Matlab/'
#pathName = "C:/Users/jhel/" #'Documents/MATLAB/wind_Matlab/'
#pathName = "/home/AD.NORCERESEARCH.NO/esje/Programs/GitHub/windpark-optimization/data/"


# Set number of quadrature points for the regions 1 ([U_cut_in, U_cut_out]) and 2 ([U_cut_out, infty])
# in speed (sp) and direction (dir)
Nq_sp = np.array([5,3])
Nq_dir = np.array([10,10])    

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
RR = rotorRadius

xmin = 0 #-4500 #-5000 #-41.6667*R0[0]#-375 #-250
xmax = 2400 #4500 #5000 #41.6667*R0[0]#-375 #250
ymin = 0 #-2500 #-3000 #-25*R0[0]#-250
ymax = 2000 #2500 #3000#25*R0[0]#250

xL = xmax - xmin - 4*RR 
yL = ymax - ymin -4*RR



# Instantiate copula object from precomputed model saved to file

copula, data_samples = copulaGeneration()

print(copula)



cop_evals_physical, wts_2D = copulaDataGeneration(copula, data_samples, Nq_sp, Nq_dir, U_cut_in, U_cut_out, U_stop)

Utmp = cop_evals_physical[0,:]
wind_dir_tmp = cop_evals_physical[1,:] #*np.pi/180. + np.pi

#num_samples = 10000 #Number of model evaluations
#num_samples = 26304
#num_dim = 2

# Some physical/technical parameters must be introduced already here
# Induction factor alpha, based on the actuator disc model
alpha = 0 #0.3333333
# Note that alpha=(U-U_R)/U, so the actual wind speeds upstream and downstream of the disc should be used


wake_model_param = np.array([C_p, rho, U_cut_in, U_cut_out, U_stop])



print("-------------------------------------------------------")
print('start time: ', now)
print("-------------------------------------------------------")




# JAN 2023, try opt under unc.


# Coordinates of all wind turbines

x_all = np.array([[0.,0.]])

x_all = np.array([[xmin+2*RR, xmin+2*RR+yL/2],[ymin+2*RR+xL, ymin+2*RR+yL/2]])

#x_all = np.array([[0,0],[100,100],[100,-100],[-100,100],[-100,-100]])
#x_all = np.array([[0,0],[100,100],[100,-100],[-100,100],[-100,-100],[0,200],[0,-200],[200,0],[-200,0]])
#x_all = np.array([[0,0],[20,20],[20,-20],[-20,20],[-20,-20],[0,40],[0,-40],[40,0],[-40,0],[40,40],[40,-40],[-40,40],[-40,-40]])
#x_all = np.array([[0,0],[20,20],[20,-20],[-20,20],[-20,-20],[0,40],[0,-40],[40,0],[-40,0],[40,40],[40,-40],[-40,40],[-40,-40], [60,20],[60,-20],[-60,20],[-60,-20]])

#3
#x_all = np.array([[2*RR, 2000-2*RR],[1200, 1000], [2400-2*RR, 2*RR]])

#4
#x_all = np.array([[xmin+2*RR, ymin+2*RR], [xmin+2*RR + xL, ymin+2*RR], [xmin+2*RR, ymin+2*RR+yL], [xmin+2*RR + xL, ymin+2*RR+yL]])

#5
#x_all = np.array([[xmin+2*RR, ymax-2*RR],[0.5*(xmin+xmax), 0.5*(ymin+ymax)], [xmax-2*RR, ymin+2*RR], [xmin+2*RR, ymin+2*RR], [xmax-2*RR, ymax-2*RR]])

#9
#x_all = np.array([[xmin+2*RR, ymax-2*RR],[0.5*(xmin+xmax), ymin+2*RR],[xmin+2*RR, 0.5*(ymin+ymax)],[xmax-2*RR, 0.5*(ymin+ymax)],[0.5*(xmin+xmax), 0.5*(ymin+ymax)], [xmax-2*RR, ymin+2*RR], [xmin+2*RR, ymin+2*RR], [xmax-2*RR, ymax-2*RR], [0.5*(xmin+xmax), ymax-2*RR]])

#12
#x_all = np.array([[xmin+2*RR, ymin+2*RR], [xmin+2*RR + xL/3, ymin+2*RR], [xmin+2*RR + 2*xL/3, ymin+2*RR], [xmin+2*RR+xL, ymin+2*RR], [xmin+2*RR, ymin+2*RR+yL/2], [xmin+2*RR + xL/3, ymin+2*RR+yL/2], [xmin+2*RR + 2*xL/3, ymin+2*RR+yL/2], [xmin+2*RR+xL, ymin+2*RR+yL/2], [xmin+2*RR, ymin+2*RR+yL], [xmin+2*RR + xL/3, ymin+2*RR+yL], [xmin+2*RR + 2*xL/3, ymin+2*RR+yL], [xmin+2*RR+xL, ymin+2*RR+yL]])
#15
#x_all = np.array([[xmin+2*RR, ymin+2*RR], [xmin+2*RR + xL/4, ymin+2*RR], [xmin+2*RR + 2*xL/4, ymin+2*RR], [xmin+2*RR+3*xL/4, ymin+2*RR], [xmin+2*RR+xL, ymin+2*RR], [xmin+2*RR, ymin+2*RR+yL/2], [xmin+2*RR + xL/4, ymin+2*RR+yL/2], [xmin+2*RR + 2*xL/4, ymin+2*RR+yL/2], [xmin+2*RR+3*xL/4, ymin+2*RR+yL/2], [xmin+2*RR+xL, ymin+2*RR+yL/2], [xmin+2*RR, ymin+2*RR+yL], [xmin+2*RR + xL/4, ymin+2*RR+yL], [xmin+2*RR + 2*xL/4, ymin+2*RR+yL], [xmin+2*RR+3*xL/4, ymin+2*RR+yL], [xmin+2*RR+xL, ymin+2*RR+yL]])

#18
#x_all = np.array([[xmin+2*RR, ymin+2*RR], [xmin+2*RR + xL/5, ymin+2*RR], [xmin+2*RR + 2*xL/5, ymin+2*RR], [xmin+2*RR+3*xL/5, ymin+2*RR], [xmin+2*RR+4*xL/5, ymin+2*RR], [xmin+2*RR+xL, ymin+2*RR], 
#                  [xmin+2*RR, ymin+2*RR+yL/2], [xmin+2*RR + xL/5, ymin+2*RR+yL/2], [xmin+2*RR + 2*xL/5, ymin+2*RR+yL/2], [xmin+2*RR+3*xL/5, ymin+2*RR+yL/2], [xmin+2*RR+4*xL/5, ymin+2*RR+yL/2], [xmin+2*RR+xL, ymin+2*RR+yL/2], 
#                  [xmin+2*RR, ymin+2*RR+yL], [xmin+2*RR + xL/5, ymin+2*RR+yL], [xmin+2*RR + 2*xL/5, ymin+2*RR+yL], [xmin+2*RR+3*xL/5, ymin+2*RR+yL], [xmin+2*RR+4*xL/5, ymin+2*RR+yL], [xmin+2*RR+xL, ymin+2*RR+yL]])

#20
#x_all = np.array([[xmin+2*RR, ymin+2*RR], [xmin+2*RR + xL/4, ymin+2*RR], [xmin+2*RR + 2*xL/4, ymin+2*RR], [xmin+2*RR+3*xL/4, ymin+2*RR], [xmin+2*RR+4*xL/4, ymin+2*RR], 
#                  [xmin+2*RR, ymin+2*RR+yL/3], [xmin+2*RR + xL/4, ymin+2*RR+yL/3], [xmin+2*RR + 2*xL/4, ymin+2*RR+yL/3], [xmin+2*RR+3*xL/4, ymin+2*RR+yL/3], [xmin+2*RR+4*xL/4, ymin+2*RR+yL/3],
#                  [xmin+2*RR, ymin+2*RR+2*yL/3], [xmin+2*RR + xL/4, ymin+2*RR+2*yL/3], [xmin+2*RR + 2*xL/4, ymin+2*RR+2*yL/3], [xmin+2*RR+3*xL/4, ymin+2*RR+2*yL/3], [xmin+2*RR+4*xL/4, ymin+2*RR+2*yL/3],   
#                  [xmin+2*RR, ymin+2*RR+yL], [xmin+2*RR + xL/4, ymin+2*RR+yL], [xmin+2*RR + 2*xL/4, ymin+2*RR+yL], [xmin+2*RR+3*xL/4, ymin+2*RR+yL], [xmin+2*RR+4*xL/4, ymin+2*RR+yL]])

#30
#x_all = np.array([[xmin+2*RR, ymin+2*RR], [xmin+2*RR + xL/5, ymin+2*RR], [xmin+2*RR + 2*xL/5, ymin+2*RR], [xmin+2*RR+3*xL/5, ymin+2*RR], [xmin+2*RR+4*xL/5, ymin+2*RR], [xmin+2*RR+5*xL/5, ymin+2*RR], 
#                  [xmin+2*RR, ymin+2*RR+yL/4], [xmin+2*RR + xL/5, ymin+2*RR+yL/4], [xmin+2*RR + 2*xL/5, ymin+2*RR+yL/4], [xmin+2*RR+3*xL/5, ymin+2*RR+yL/4], [xmin+2*RR+4*xL/5, ymin+2*RR+yL/4], [xmin+2*RR+5*xL/5, ymin+2*RR+yL/4],
#                  [xmin+2*RR, ymin+2*RR+2*yL/4], [xmin+2*RR + xL/5, ymin+2*RR+2*yL/4], [xmin+2*RR + 2*xL/5, ymin+2*RR+2*yL/4], [xmin+2*RR+3*xL/5, ymin+2*RR+2*yL/4], [xmin+2*RR+4*xL/5, ymin+2*RR+2*yL/4], [xmin+2*RR+5*xL/5, ymin+2*RR+2*yL/4], 
#                  [xmin+2*RR, ymin+2*RR+3*yL/4], [xmin+2*RR + xL/5, ymin+2*RR+3*yL/4], [xmin+2*RR + 2*xL/5, ymin+2*RR+3*yL/4], [xmin+2*RR+3*xL/5, ymin+2*RR+3*yL/4], [xmin+2*RR+4*xL/5, ymin+2*RR+3*yL/4], [xmin+2*RR+5*xL/5, ymin+2*RR+3*yL/4],
#                  [xmin+2*RR, ymin+2*RR+4*yL/4], [xmin+2*RR + xL/5, ymin+2*RR+4*yL/4], [xmin+2*RR + 2*xL/5, ymin+2*RR+4*yL/4], [xmin+2*RR+3*xL/5, ymin+2*RR+4*yL/4], [xmin+2*RR+4*xL/5, ymin+2*RR+4*yL/4], [xmin+2*RR+5*xL/5, ymin+2*RR+4*yL/4]])



#x_all = np.array([[0, 0],[-500, 0], [500, 0],[-1000, 0], [1000, 0]])

#x_all = np.array([[0, 0],[-500, 0], [500, 0],[-1000, 0], [1000, 0], [-2000, 0], [2000, 0], [3000, 0]])

#x_all = np.array([[-3e3, -1e3], [-1e3, -1e3], [1e3,-1e3], [3e3, -1e3], [-3e3, 5e2], [-1e3, 5e2], [1e3, 5e2], [3e3, 5e2], [-3e3, 2e3], [-1e3, 2e3], [1e3,2e3], [3e3, 2e3]])

#x_all = np.array([[-4500, 3000], [-2250, 3000], [0,3000], [2250, 3000], [4500,3000],[-4500, 1000], [-2250, 1000], [0,1000], [2250, 1000], [4500,1000], [-4500, -1000], [-2250, -1000], [0, -1000], [2250, -1000], [4500, -1000], [-4500, -3000], [-2250, -3000], [0,-3000], [2250, -3000], [4500,-3000]])
#x_all = np.array([[-4500, 2500], [-2250, 2500], [0,2500], [2250, 2500], [4500,2500],[-4500, 1000], [-2250, 1000], [0,1000], [2250, 1000], [4500,1000]])
#x_all = np.array([[-3750, 2500], [-2250, 2500], [-750,2500], [750,2500],[2250, 2500], [750, 2500], [-3750,0], [-2250, 0], [-750, 0], [750, 0], [2250, 0], [3750, 0], [-3750, -2500], [-2250, -2500], [-750, -2500], [750, -2500], [225,-2500], [ 3750, -2500]])

#x_all = np.array([[-4500, 2500], [-3000, 2500], [-1500,2500], [0, 2500], [1500,2500],[3000, 2500], [4500, 2500], [-4500,0], [-3000, 0], [-1500, 0], [0, 0], [1500, 0], [3000, 0], [4500, 0], [-4500, -2500], [-3000, -2500], [-1500, -2500], [0,-2500], [1500, -2500], [3000,-2500], [ 4500, -2500]])



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

x_grid = np.linspace(xmin-4*R0[0], 50*R0[0], N_x)
y_grid = np.linspace(ymin-4*R0[0], 50*R0[0], N_y)

U = np.mean(cop_evals_physical[0,:])
wind_dir = np.mean(cop_evals_physical[1,:])  #*np.pi/180. + np.pi


u_eval = np.zeros((N_x, N_y))
delta_u_eval = np.zeros((N_turb, N_x, N_y))

u_eval_optim = np.zeros((N_x, N_y))
delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))

####################

# fig1 = plt.figure(constrained_layout=True) #JOH
# fig1.set_size_inches(4.2, 3.5) #JOH
# plt.scatter(x_all[:,0], x_all[:,1], s=50, c='black', marker='+')
# plt.plot([xmin, xmin], [ymin, ymax], linestyle='dashed', color='black')
# plt.plot([xmax, xmax], [ymin, ymax], linestyle='dashed', color='black')
# plt.plot([xmin, xmax], [ymin, ymin], linestyle='dashed', color='black')
# plt.plot([xmin, xmax], [ymax, ymax], linestyle='dashed', color='black')

# for x, y in zip(x_all[:,0], x_all[:,1]):
#     circle = plt.Circle((x,y), radius=R0[0], fill=False, linestyle='--', color='black')
#     fig2=plt.gcf()
#     ax=fig2.gca()
#     ax.add_patch(circle)

                        
# plt.xlabel('$X\ [\mathrm{m}]$')
# plt.ylabel('$Y\ [\mathrm{m}]$', rotation=90)
# plt.title('Initial Configuration') #('Wind speed, opt. locations')

# plt.show()
# ####################
# exit(1)

fileName = pathName+'arrays1/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.npz'
fileName2 = pathName+'arrays1/mean_dir_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.npz'

print("-----------")
print("Initial turbine Production: ")
print("-----------")
objective_fun_num_robust_design(x_vector, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p) # OLAV 
print("-----------")

obj_fun = functools.partial(objective_fun_num_robust_design, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p)

#res = minimize(obj_fun, x_vector, method='SLSQP', jac=True, options={'disp': True}, constraints=[lin_constr, nonlinear_constraint]) #constraints=lin_constr) 
res = minimize(obj_fun, x_vector, method='SLSQP', jac=True, options={'disp': True, 'ftol': opt_tolerance_SLSQP, 'maxiter': maxiter_SLSQP}, constraints=[lin_constr, nonlinear_constraint]) # default: 'ftol': 1e-06

tmp, tmp2, power_mom = objective_fun_num_robust_design(res.x, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p)


np.savez(fileName, init_pos=x_vector, end_pos=res.x, rotor_rad=R0, wake_param=wake_model_param, confine_rectangle = confinement_rectangle, power_moms = power_mom)

x_opt = np.reshape(res.x, (N_turb,2))
print("Initial turbine locations: ", x_vector)
print("Objective fun value, init: ", obj_fun(x_vector))

print("Robust design, New turbine locations: ", res.x)
 

print("Robust design, New E[power]= ", power_mom[0])
print("Robust design, New sigma[power]= ", power_mom[1])


print("Objective fun value, opt: ", res.fun)
"""
for ix in range(N_x):
    for iy in range(N_y):
        x_i = (x_grid[ix], y_grid[iy])

        u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
        delta_u_eval_optim[:,ix,iy] = temp.flatten()
"""    


(xv, yv) = np.meshgrid(x_grid, y_grid)



wt_pos = np.zeros((3, N_turb))
print("N_turb = ", N_turb)
wt_pos[0, :] = res.x[::2]
wt_pos[1, :] = res.x[1::2]
print("wt_pos:", wt_pos)
kappa = 0.05

ang = 0.5*np.pi - (wind_dir )#*np.pi/180 
vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

blah=wfm.Windfarm(wt_pos, vec_dir, R0[0], U_cut_in, U_cut_out, rho, kappa, C_p) 
blah.power(U)
u_eval_optim = blah.wind_field(U, xv, yv)

fig = plt.figure(constrained_layout=True)
fig.set_size_inches(4.0, 4.0)
#cmap2 = plt.get_cmap("jet")
im1= plt.imshow(u_eval_optim, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
#im1= plt.pcolormesh(xv,yv, u_eval_optim)
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
im1= plt.imshow(u_eval_optim, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
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
tmp, tmp2, power_mom = objective_fun_num_robust_design(res.x, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p) # OLAV 
#obj_fun = functools.partial(objective_fun_num_robust_design, pts=cop_evals_physical, wts=wts_2D, R0=R0, alpha=alpha, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, C_p=C_p)

np.savez(fileName2, init_pos=x_vector, end_pos=res.x, rotor_rad=R0, wake_param=wake_model_param, confine_rectangle = confinement_rectangle, power_moms = power_mom)

print("Initial turbine locations: ", x_vector)
print("New turbine locations, mean dir and speed: ", res.x)
print("Objective fun value, init: ", obj_fun(x_vector))
print("Objective fun value, opt: ", res.fun)
"""
for ix in range(N_x):
    for iy in range(N_y):
        x_i = (x_grid[ix], y_grid[iy])

        u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
        delta_u_eval_optim[:,ix,iy] = temp.flatten()

"""
###
(xv, yv) = np.meshgrid(x_grid, y_grid)

wt_pos = np.zeros((3, N_turb))
print("N_turb = ", N_turb)
wt_pos[0, :] = res.x[::2]
wt_pos[1, :] = res.x[1::2]
print("wt_pos:", wt_pos)
kappa = 0.05

ang = 0.5*np.pi - (wind_dir )#*np.pi/180 
vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

blah=wfm.Windfarm(wt_pos, vec_dir, R0[0], U_cut_in, U_cut_out, rho, kappa, C_p) 
blah.power(U)
u_eval_optim = blah.wind_field(U, xv, yv)



now = datetime.datetime.now()
print("-------------------------------------------------------")
print('end mean-wind opt. time: ', now)
print("-------------------------------------------------------")

#(xv, yv) = np.meshgrid(x_grid, y_grid)
fig2 = plt.figure(constrained_layout=True)
fig2.set_size_inches(4.2, 3.5)
im1= plt.imshow(u_eval_optim, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
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

