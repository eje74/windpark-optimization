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


from scipy.special import rel_entr
from pylab import *




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
    

    return u_i, delta_u_i


def repeat_product(x, y):
    #return np.transpose([np.tile(x, len(y)),
    #                        np.repeat(y, len(x))])
    return np.transpose([np.repeat(x, len(y)),
                            np.tile(y, len(x))])   


def copulaGeneration():
    theta_data  = np.loadtxt(pathName + 'Dir_100m_2016_2017_2018.txt')
    u_data  = np.loadtxt(pathName + 'Sp_100m_2016_2017_2018.txt')

    theta_data[theta_data<0] = theta_data[theta_data<0] + 360
    theta_data[theta_data>360] = theta_data[theta_data>360] - 360

    data_samples_loc = np.array([u_data, theta_data])

    # Instantiate copula object from precomputed model saved to file
    copula_loc = pv.Vinecop(filename = pathName + 'copula_tree_100m_2016_2017_2018.txt', check=True)

    return copula_loc, data_samples_loc

def windDataGeneration(copula_loc, data_samples_loc, Nq_sp_loc, Nq_dir_loc):

  

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
    p_0 = np.sum(u_data < U_cut_in)/num_samples
    ind_var_s = (u_data < U_cut_out) & (u_data >= U_cut_in)
    ind_con_s = u_data >= U_cut_out
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

    print('windDataGeneration() done')

    return cop_evals_physical[0,:], cop_evals_physical[1,:]*np.pi/180. + np.pi, wts_2D

    ######################################################################################################

def windDataGenerationPlot(copula_loc, data_samples_loc, Nq_sp_plot, Nq_dir_plot):

    num_dim = data_samples_loc.shape[0]

    [pts_sp_plot, wts_sp_plot] = np.polynomial.legendre.leggauss(Nq_sp_plot)
    [pts_dir_plot, wts_dir_plot] = np.polynomial.legendre.leggauss(Nq_dir_plot)

    # Rescale to unit interval, and weights summing to 1
    pts_sp_plot = (pts_sp_plot+1)/2.
    wts_sp_plot = 0.5*wts_sp_plot

    pts_dir_plot = (pts_dir_plot+1)/2.
    wts_dir_plot = 0.5*wts_dir_plot

    #print("Quadrature tests 1D: ", sum(wts), sum(pts*wts), sum((pts**2)*wts), sum(pts**3*wts))
    # Quadrature for U in [U_cut_in, U_cut_out]
    pts_2D_plot = repeat_product(pts_sp_plot, pts_dir_plot).T
    wts_2D_plot = np.kron(wts_sp_plot, wts_dir_plot)

    #print("pts_2D_plot: ", pts_2D_plot)
    #print("pts_2D: ", pts_2D)

    # Evaluate copula via inverse Rosenblatt
    cop_evals_plot = copula_loc.inverse_rosenblatt(pts_2D_plot.T)

    # Transform back evaluations to the original scale
    cop_evals_physical_plot = np.asarray([np.quantile(data_samples_loc[i,:], cop_evals_plot[:, i]) for i in range(0, num_dim)])



    

    #######################################################################################################

    print('windDataGenerationPlot() done')

    return cop_evals_physical_plot[0,:], cop_evals_physical_plot[1,:]*np.pi/180. + np.pi, wts_2D_plot


# END function definitions

####################################################################################
################################### MAIN ###########################################

plt.close('all')
plt.rcParams['text.usetex'] = True


#NBNBNBNBNBNBNB

alpha = 1/3.

#NBNBNBNBNBNBNB

#
N_turb = 2

Nq_sp = np.array([5,3])
Nq_dir = np.array([10,10])


Nq_sp_plot = 5#10
Nq_dir_plot = 20#15 

VminPlot = 8
VmaxPlot = 10


print("test: ", str(Nq_sp) )

pathName = '/home/AD.NORCERESEARCH.NO/olau/Documents/projects/DynPosWind/opt_farm/data/'
fileName = pathName+'arrays1/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+ str(Nq_dir) + '.npz'

npzfile = np.load(fileName)
print('Tag names of saved arrays:', npzfile.files)


copula, data_samples = copulaGeneration()



init_pos = npzfile['init_pos']
end_pos = npzfile['end_pos']
wake_model_param = npzfile['wake_param']
R0 = npzfile['rotor_rad']
confinement_rectangle = npzfile['confine_rectangle']

#------------------------Wake model parameters----------------------------------
C_p = wake_model_param[0]
rho =  wake_model_param[1]
U_cut_in = wake_model_param[2]
U_cut_out = wake_model_param[3]

xmin = confinement_rectangle[0]
ymin = confinement_rectangle[1]
xmax = confinement_rectangle[2]
ymax = confinement_rectangle[3]

print('Initial turbine locations:', init_pos)
print('New turbine locations:', end_pos)
print('R0 =', R0)
print('[C_p, rho, U_cut_in, U_cut_out]', wake_model_param)
print('[xmin ymin xmax ymax]', confinement_rectangle)


print("max power each turbine, P=", 0.5*rho*np.pi*R0**2*C_p*U_cut_out**3/1e6,"MW")





# Visualize wind field due to wake effects
N_x = 100
N_y = 100
x_grid = np.linspace(-50*R0[0], 50*R0[0], N_x)
y_grid = np.linspace(-50*R0[0], 50*R0[0], N_y)

Utmp, wind_dir_tmp, _ = windDataGeneration(copula, data_samples, Nq_sp, Nq_dir)
U = np.mean(Utmp)
wind_dir = np.mean(wind_dir_tmp)

u_eval = np.zeros((N_x, N_y))
delta_u_eval = np.zeros((N_turb, N_x, N_y))

u_eval_optim = np.zeros((N_x, N_y))
delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))

D = 2*R0

x_opt = np.reshape(end_pos, (N_turb,2))

for ix in range(N_x):
    for iy in range(N_y):
        x_i = (x_grid[ix], y_grid[iy])

        u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, x_opt, U, wind_dir, D, r_i = 0, theta_i = 0)
    


(xv, yv) = np.meshgrid(x_grid, y_grid)


fig1 = plt.figure(constrained_layout=True) 
fig1.set_size_inches(4.2, 3.5) 
im1= plt.imshow(u_eval_optim.T, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
#cset = plt.contourf(xv, yv, u_eval_optim.T, cmap=cm.coolwarm)
#plt.scatter(x_all[:,0], x_all[:,1], s=50, c='black', marker='+')
#plt.scatter(x_opt[:,0], x_opt[:,1], s=30, c='black', marker='o')
plt.plot([xmin, xmin], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmax, xmax], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymin, ymin], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymax, ymax], linestyle='dashed', color='black')

fig=plt.gcf()
ax=fig.gca()

#for x, y in zip(x_opt[:,0], x_opt[:,1]):
#    circle = plt.Circle((x,y), radius=R0[0], fill=False, linestyle='--', color='black')
#    fig=plt.gcf()
#    ax=fig.gca()
#    ax.add_patch(circle)
   
                                      

ax.set_aspect('equal') #JOH
fig.colorbar(im1)
plt.xlabel('$X\ [\mathrm{m}]$')
plt.ylabel('$Y\ [\mathrm{m}]$', rotation=90)
plt.title('Robust design') #('Wind speed, opt. locations')



U, wind_dir, wts_2D_plot  = windDataGenerationPlot(copula, data_samples, Nq_sp_plot, Nq_dir_plot)


u_eval_optim *=0

for ix in range(N_x):
    print('ix=', ix)
    for iy in range(N_y):
        
        x_i = (x_grid[ix], y_grid[iy])



        for q in range(Nq_sp_plot*Nq_dir_plot):
           
            #U = cop_evals_physical_plot[0,q]
            #wind_dir = cop_evals_physical_plot[1,q]*np.pi/180.
            
            #u_eval_optim[ix,iy], temp = wind_speed_due_to_wake(x_i, np.reshape(res.x, (N_turb,2)), U, wind_dir, D, r_i = 0, theta_i = 0)
            temp_u, temp_delta = wind_speed_due_to_wake(x_i, x_opt, U[q], wind_dir[q], D, r_i = 0, theta_i = 0)
            
            u_eval_optim[ix,iy] += temp_u*wts_2D_plot[q]
            delta_u_eval_optim[:,ix,iy] += wts_2D_plot[q]*temp_delta.flatten()
    

(xv, yv) = np.meshgrid(x_grid, y_grid)

fileName = pathName+'arrays1/wake_mean' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.npz'
np.savez(fileName, u_eval=u_eval_optim, xv=xv, yv=yv)


fig = plt.figure(constrained_layout=True)
fig.set_size_inches(4.0, 4.0)
#cmap2 = plt.get_cmap("jet")

###

#bounds=[4,5,6,7,8,10]

#cmap = colors.ListedColormap(plt.cm.coolwarm)
#norm = colors.BoundaryNorm(bounds, cmap.N)
###

im1= plt.imshow(u_eval_optim.T, interpolation='nearest', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, vmin=VminPlot, vmax=VmaxPlot, origin='lower', aspect='auto')
# interpolation='bicubic' 'antialiased'
#plt.scatter(x_all[:,0], x_all[:,1], s=50, c='blue', marker='+')
fig.colorbar(im1)
plt.xlabel('$X$ [m]')
plt.ylabel('$Y$ [m]', rotation=0)


#cmap2 = plt.get_cmap("jet")

###

#bounds=[4,5,6,7,8,10]

#cmap = colors.ListedColormap(plt.cm.coolwarm)
#norm = colors.BoundaryNorm(bounds, cmap.N)
##

###
###
fig1 = plt.figure(constrained_layout=True) #JOH
fig1.set_size_inches(4.2, 3.5) #JOH
cmap=plt.cm.coolwarm #([0,0.1,0.2,0.4,0.7,1])
figLevels = np.linspace(VminPlot, VmaxPlot, 12)
# [4,5,6,7,8,9.5] [0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.95]
im1= plt.imshow(u_eval_optim.T, interpolation='nearest', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, vmin=VminPlot, vmax=VmaxPlot, origin='lower', aspect='auto')
cset = plt.contourf(xv, yv, u_eval_optim.T, levels=figLevels, cmap=cmap)
#plt.scatter(x_all[:,0], x_all[:,1], s=50, c='black', marker='+')
#plt.scatter(x_opt[:,0], x_opt[:,1], s=30, c='black', marker='o')
plt.plot([xmin, xmin], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmax, xmax], [ymin, ymax], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymin, ymin], linestyle='dashed', color='black')
plt.plot([xmin, xmax], [ymax, ymax], linestyle='dashed', color='black')



fig=plt.gcf()
ax=fig.gca()

#for x, y in zip(x_opt[:,0], x_opt[:,1]):
#    circle = plt.Circle((x,y), radius=R0[0], fill=False, linestyle='--', color='black')
#    fig=plt.gcf()
#    ax=fig.gca()
#    ax.add_patch(circle)
   
                                      

ax.set_aspect('equal') #JOH
fig.colorbar(im1)
plt.xlabel('$X\ [\mathrm{m}]$')
plt.ylabel('$Y\ [\mathrm{m}]$', rotation=90)
plt.title('Robust design') #('Wind speed, opt. locations')
















plt.show()




