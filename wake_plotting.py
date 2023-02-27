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

def repeat_product(x, y):
    #return np.transpose([np.tile(x, len(y)),
    #                        np.repeat(y, len(x))])
    return np.transpose([np.repeat(x, len(y)),
                            np.tile(y, len(x))])   


###################################################################

def objective_fun_num_robust_design(x_vector, pts, wts, R0_loc, rho, U_cut_in, U_cut_out, U_stop, C_p):
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

    ang = 0.5*np.pi - wind_dir
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
    

###################################################################

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



# END function definitions

####################################################################################
################################### MAIN ###########################################

plt.close('all')
plt.rcParams['text.usetex'] = True


#NBNBNBNBNBNBNB

#alpha = 0.

#NBNBNBNBNBNBNB

#
N_turb = 2

Nq_sp = np.array([5,3])
Nq_dir = np.array([10,10])


Nq_sp_plot = 15
Nq_dir_plot = 20#15 

VminPlot = 9 #8
VmaxPlot = 10


print("test: ", str(Nq_sp) )

pathName = '/home/AD.NORCERESEARCH.NO/olau/Documents/projects/DynPosWind/opt_farm/data/'
fileName = pathName+'arrays1/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+ str(Nq_dir) + '.npz'
fileName2 = pathName+'arrays1/mean_dir_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+ str(Nq_dir) + '.npz'
npzfile = np.load(fileName)
npzfile2 = np.load(fileName2)
print('Tag names of saved arrays:', npzfile.files)


copula, data_samples = copulaGeneration()



init_pos = npzfile['init_pos']
end_pos = npzfile['end_pos']
wake_model_param = npzfile['wake_param']
R0 = npzfile['rotor_rad']
confinement_rectangle = npzfile['confine_rectangle']
power_moms = npzfile['power_moms']


init_pos_mean = npzfile2['init_pos']
end_pos_mean = npzfile2['end_pos']
power_moms_mean = npzfile2['power_moms']

#------------------------Wake model parameters----------------------------------
C_p = wake_model_param[0]
rho =  wake_model_param[1]
U_cut_in = wake_model_param[2]
U_cut_out = wake_model_param[3]
U_stop = wake_model_param[4]

xmin = confinement_rectangle[0]
ymin = confinement_rectangle[1]
xmax = confinement_rectangle[2]
ymax = confinement_rectangle[3]

print('Initial turbine locations:', init_pos)
print('New turbine locations:', end_pos)
print('R0 =', R0)
print('[C_p, rho, U_cut_in, U_cut_out, U_stop]', wake_model_param)
print('[xmin ymin xmax ymax]', confinement_rectangle)


print("max power each turbine, P=", 0.5*rho*np.pi*R0**2*C_p*U_cut_out**3/1e6,"MW")
print("------------------------------------")
print("Robust design, New E[power]= ", -power_moms[0])
print("Robust design, New sigma[power]= ", power_moms[1])
print("------------------------------------")

print("Mean wind, New E[power]= ", -power_moms_mean[0])
print("------------------------------------")



# Visualize wind field due to wake effects
N_x = 300
N_y = 300
x_grid = np.linspace(-50*R0[0], 50*R0[0], N_x)
y_grid = np.linspace(-50*R0[0], 50*R0[0], N_y)

#Utmp, wind_dir_tmp, _ = windDataGeneration(copula, data_samples, Nq_sp, Nq_dir)


cop_evals_physical, wts_2D = copulaDataGeneration(copula, data_samples, Nq_sp, Nq_dir, U_cut_in, U_cut_out, U_stop)

Utmp = cop_evals_physical[0,:]
wind_dir_tmp = cop_evals_physical[1,:]

print("wind Directions in radians:", wind_dir_tmp)


U = np.mean(Utmp)
wind_dir =  np.mean(wind_dir_tmp)

u_eval = np.zeros((N_x, N_y))
delta_u_eval = np.zeros((N_turb, N_x, N_y))

u_eval_optim = np.zeros((N_x, N_y))
delta_u_eval_optim = np.zeros((N_turb, N_x, N_y))

D = 2*R0


(xv, yv) = np.meshgrid(x_grid, y_grid)


wt_pos = np.zeros((3, N_turb))
print("N_turb = ", N_turb)
wt_pos[0, :] = end_pos[::2]
wt_pos[1, :] = end_pos[1::2]
print("wt_pos:", wt_pos)
kappa = 0.05

ang = 0.5*np.pi - (wind_dir )#*np.pi/180 
vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

blah=wfm.Windfarm(wt_pos, vec_dir, R0[0], U_cut_in, U_cut_out, rho, kappa, C_p) 
print("Power:", blah.power(U))
u_eval_optim = blah.wind_field(U, xv, yv)


fig1 = plt.figure(constrained_layout=True) 
fig1.set_size_inches(4.2, 3.5) 
im1= plt.imshow(u_eval_optim, interpolation='antialiased', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, origin='lower', aspect='auto')
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


mu=0

for q in range(len(wts_2D)):
    ang = 0.5*np.pi - (wind_dir_tmp[q] )#*np.pi/180 
    vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

    blah=wfm.Windfarm(wt_pos, vec_dir, R0[0], U_cut_in, U_cut_out, rho, kappa, C_p) 
    P_evals=blah.power(Utmp[q])
    u_eval_optim += blah.wind_field(Utmp[q], xv, yv)*wts_2D[q]
    mu += P_evals*wts_2D[q]

print("------------------------------------")
print("Plotting Total Power:", mu)
print("------------------------------------")


tmp, tmp2, power_mom = objective_fun_num_robust_design(end_pos, pts=cop_evals_physical, wts=wts_2D, R0_loc=R0, rho=rho, U_cut_in=U_cut_in, U_cut_out=U_cut_out, U_stop=U_stop, C_p=C_p)

print("Plotting obj Power:", power_mom[0])

U, wind_dir, wts_2D_plot  = windDataGenerationPlot(copula, data_samples, Nq_sp_plot, Nq_dir_plot)

#U = cop_evals_physical[0,:]
#wind_dir = cop_evals_physical[1,:]*np.pi/180. + np.pi
#wts_2D_plot = wts_2D

u_eval_optim *=0



(xv, yv) = np.meshgrid(x_grid, y_grid)

for q in range(len(wts_2D_plot)):
    ang = 0.5*np.pi - (wind_dir[q] )#*np.pi/180 
    vec_dir = np.array([np.cos(ang), np.sin(ang), 0])

    blah=wfm.Windfarm(wt_pos, vec_dir, R0[0], U_cut_in, U_cut_out, rho, kappa, C_p) 
    P_evals=blah.power(U[q])
    u_eval_optim += blah.wind_field(U[q], xv, yv)*wts_2D_plot[q]
    mu += P_evals*wts_2D_plot[q]


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

im1= plt.imshow(u_eval_optim, interpolation='nearest', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, vmin=VminPlot, vmax=VmaxPlot, origin='lower', aspect='auto')
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
#figLevels =  [4,5,6,7,8,9.5] #[0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.95]
im1= plt.imshow(u_eval_optim, interpolation='nearest', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, vmin=VminPlot, vmax=VmaxPlot, origin='lower', aspect='auto')
cset = plt.contourf(xv, yv, u_eval_optim, levels=figLevels, cmap=cmap)

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




