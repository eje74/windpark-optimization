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





####################################################################################
################################### MAIN ###########################################

plt.close('all')
plt.rcParams['text.usetex'] = True




#
N_turb = 2

Nq_sp = np.array([5,3])
Nq_dir = np.array([10,10])

VminPlot = 8
VmaxPlot = 10


print("test: ", str(Nq_sp) )

pathName = '/home/AD.NORCERESEARCH.NO/olau/Documents/projects/DynPosWind/opt_farm/data/'
fileName = pathName+'arrays1/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+ str(Nq_dir) + '.npz'

npzfile = np.load(fileName)
print('Tag names of saved arrays:', npzfile.files)

init_pos = npzfile['init_pos']
end_pos = npzfile['end_pos']
R0 = npzfile['rotor_rad']
confinement_rectangle = npzfile['confine_rectangle']

x_init = np.reshape(init_pos, (N_turb,2))
x_opt = np.reshape(end_pos, (N_turb,2))

xmin = confinement_rectangle[0]
ymin = confinement_rectangle[1]
xmax = confinement_rectangle[2]
ymax = confinement_rectangle[3]

fileName2 = pathName+'arrays1/wake_mean' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.npz'


npzfile = np.load(fileName2)
print('Tag names of saved arrays:', npzfile.files)

u_eval_optim = npzfile['u_eval']
xv = npzfile['xv']
yv = npzfile['yv']



###
###
fig1 = plt.figure(constrained_layout=True) 
fig1.set_size_inches(4.2*1.3, 3.5*1.3) 
cmap=plt.cm.coolwarm #([0,0.1,0.2,0.4,0.7,1])
figLevels = np.linspace(VminPlot, VmaxPlot, 12)
im1= plt.imshow(u_eval_optim.T, interpolation='nearest', extent=(xv.min(),xv.max(),yv.min(),yv.max()), cmap=plt.cm.coolwarm, vmin=VminPlot, vmax=VmaxPlot, origin='lower', aspect='auto')

#plt.pcolormesh((rhoUni[:,:,0].transpose()-rhoVar[:,:,0].transpose())/1e-3, cmap=cm, rasterized=True)

cset = plt.contourf(xv, yv, u_eval_optim.T, levels=figLevels, cmap=cmap)
plt.scatter(x_init[:,0], x_init[:,1], s=30, c='black', linewidths=1.0, marker='+')
#plt.scatter(x_opt[:,0], x_opt[:,1], c='black', linewidths=1.0, marker='o')
plt.plot([xmin, xmin], [ymin, ymax], linestyle='dashed', linewidth=1.0, color='black')
plt.plot([xmax, xmax], [ymin, ymax], linestyle='dashed', linewidth=1.0, color='black')
plt.plot([xmin, xmax], [ymin, ymin], linestyle='dashed', linewidth=1.0, color='black')
plt.plot([xmin, xmax], [ymax, ymax], linestyle='dashed', linewidth=1.0, color='black')



fig=plt.gcf()
ax=fig.gca()

for x, y in zip(x_opt[:,0], x_opt[:,1]):
    circle = plt.Circle((x,y), radius=R0[0], fill=False, linewidth=1.0, linestyle='--', color='black')
    
    fig=plt.gcf()
    ax=fig.gca()
    ax.add_patch(circle)
for x, y in zip(x_opt[:,0], x_opt[:,1]):
    
    circle = plt.Circle((x,y), radius=0.2*R0[0], fill=True, linewidth=1.0, linestyle='-', color='black')
    fig=plt.gcf()
    ax=fig.gca()
    ax.add_patch(circle)   
                                      

ax.set_aspect('equal') #JOH
fig.colorbar(im1)
plt.xlabel('$X\ [\mathrm{m}]$')
plt.ylabel('$Y\ [\mathrm{m}]$', rotation=90)
plt.title('Robust design') #('Wind speed, opt. locations')


fignameOut = pathName+'figures3/Robust_design_Nturb_' + str(N_turb)  + '_Nq(sp,dir)_' + str(Nq_sp) +'_'+str(Nq_dir) + '.pdf'

plt.savefig(fignameOut, bbox_inches = "tight", transparent = "True")













plt.show()




