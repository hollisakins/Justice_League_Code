import astropy.units as u
from astropy.cosmology import Planck15, z_at_value
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import tqdm

mpl.rc('font',**{'family':'serif','monospace':['Palatino']})
mpl.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = False
mpl.rcParams.update({'font.size': 9})


## specify path to data
path = '../../Data/tracked_particles.hdf5'
print('Loading data')
## load in the data
# key = 'h242_24'
key = str(sys.argv[1])
data = pd.read_hdf(path, key=key)
data

R = 1.5
N = 50

fig = plt.figure(dpi=300, figsize=(7.5,4))
gs = mpl.gridspec.GridSpec(2,5, width_ratios = [1,0.1,1,1,0.15], figure=fig)
ax0 = plt.subplot(gs[0,0])
ax1 = plt.subplot(gs[1,0])
ax2 = plt.subplot(gs[0:,2:-1])
cbax = plt.subplot(gs[:,-1])

host_radius = plt.Circle((0, 0), 1, color='k', fill=False)
ax0.add_artist(host_radius)

host_radius = plt.Circle((0, 0), 1, color='k', fill=False)
ax1.add_artist(host_radius)

host_radius = plt.Circle((0, 0), 1, color='k', fill=False)
ax2.add_artist(host_radius)

x_rel, y_rel, z_rel, Rvirs, ts = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

print('Plotting satellite orbits...')
for t in tqdm.tqdm(np.unique(data.time)):
    d = data[data.time==t]
    sat_x, sat_y, sat_z = np.mean(d.sat_Xc), np.mean(d.sat_Yc), np.mean(d.sat_Zc)
    host_x, host_y, host_z = np.mean(d.host_Xc), np.mean(d.host_Yc), np.mean(d.host_Zc)
    
    h1Rvir = np.mean(d.h1Rvir)
    satRvir = np.mean(d.satRvir)
    
    x_rel = np.append(x_rel, (sat_x-host_x)/h1Rvir)
    y_rel = np.append(y_rel, (sat_y-host_y)/h1Rvir)
    z_rel = np.append(z_rel, (sat_z-host_z)/h1Rvir)
    
    Rvirs = np.append(Rvirs, satRvir/h1Rvir*1100)
    
    radii = plt.Circle(((sat_x-host_x)/h1Rvir,(sat_y-host_y)/h1Rvir),
                     satRvir/h1Rvir, ec='none',fc='mistyrose', alpha=0.5, fill=True, zorder=0.5)
    ax0.add_artist(radii)
    
    radii = plt.Circle(((sat_x-host_x)/h1Rvir,(sat_z-host_z)/h1Rvir),
                     satRvir/h1Rvir, ec='none',fc='mistyrose', alpha=0.5, fill=True, zorder=0.5)
    ax1.add_artist(radii)
    
    ts = np.append(ts,t)
    
ax0.scatter(x_rel, y_rel, c=ts, cmap='Reds', s=4, zorder=2)
ax1.scatter(x_rel, z_rel, c=ts, cmap='Reds', s=4, zorder=2)
ax2.plot(x_rel, y_rel, 'k--', linewidth=1, zorder=1)
ax2.plot(x_rel, y_rel, 'k--', linewidth=1, zorder=1)
ax2.plot(x_rel, y_rel, 'k--', linewidth=1, zorder=1000)

pids = np.unique(data.pid)
np.random.seed(123)
pids_sub = np.random.choice(pids, size=N)

print('Plotting particle paths...')
for j,pid in enumerate(tqdm.tqdm(pids_sub)):
    d = data[data.pid==pid]
    
    cmap = mpl.cm.get_cmap('Reds', 12)
    tmin, tmax = np.min(d.time), np.max(d.time)

    ### hack for now, until I can go back into the code and scale xyz by a taken directly from the sim
    z = np.array([z_at_value(Planck15.age, (t-0.01)*u.Gyr) for t in d.time])
    a = 1/(1+z)
    
    ###
    
    i_prev = 0
    for i in range(len(d)-1):
        i += 1
        h1Rvir = list(d.h1Rvir)[i_prev]
        x1 = list(d.x_rel_host)[i_prev] / h1Rvir / a[i_prev]
        y1 = list(d.y_rel_host)[i_prev] / h1Rvir / a[i_prev]
        z1 = list(d.z_rel_host)[i_prev] / h1Rvir / a[i_prev]
        
        h1Rvir = list(d.h1Rvir)[i]
        x2 = list(d.x_rel_host)[i] / h1Rvir / a[i]
        y2 = list(d.y_rel_host)[i] / h1Rvir / a[i]
        z2 = list(d.z_rel_host)[i] / h1Rvir / a[i]
        t = list(d.time)[i]
        
        c = cmap((t-tmin)/(tmax-tmin))
                
        ax2.plot([x1,x2],[y1,y2],color=c, linewidth=1, zorder=j)
        
        #if j == 0:
        #    ax.arrow(0, 0, v_rel[0], v_rel[1], head_width=0.3, head_length=0.3, fc=c, ec=c, zorder=100+j)

        i_prev = i
    
print('Finalizing and saving figure...')

cb1 = mpl.colorbar.ColorbarBase(cbax, cmap=mpl.cm.Reds, orientation='vertical', 
                                norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax),
                                label='Time [Gyr]')

ax1.set_xlabel(r'$x$ [host $R_{\rm vir}$]')
ax1.set_ylabel(r'$z$ [host $R_{\rm vir}$]')
ax0.set_ylabel(r'$y$ [host $R_{\rm vir}$]')
# ax0.annotate(f"{key.replace('_','-')}, satellite orbit", (0.04, 0.96), xycoords='axes fraction', va='top')
ax2.annotate(f"{key.replace('_','-')}, tracked particles", (0.04, 0.96), xycoords='axes fraction', va='top')
ax2.annotate(r'$N_{\rm tot} =$' + fr' ${len(pids)}$'+'\n'+r'$N_{\rm sub} =$' + fr' ${N}$', 
            (0.96, 0.96), xycoords='axes fraction', va='top', ha='right')

ax0.set_xlim(-R,R)
ax0.set_ylim(-R,R)
ax1.set_xlim(-R,R)
ax1.set_ylim(-R,R)
ax2.set_xlim(-R,R)
ax2.set_ylim(-R,R)

ax0.set_aspect('equal')
ax1.set_aspect('equal')
ax2.set_aspect('equal')

# ax2.set_xlabel(r'rotated coordinate $\tilde{x}$ [sat $R_{\rm vir}$]')
# ax2.set_ylabel(r'rotated coordinate $\tilde{y}$ [sat $R_{\rm vir}$]')
ax2.set_xlabel(r'$x$ [host $R_{\rm vir}$]')
ax2.set_ylabel(r'$y$ [host $R_{\rm vir}$]')


ax0.tick_params(top=True,right=True,direction='in', which='both')
ax1.tick_params(top=True,right=True,direction='in', which='both')
ax2.tick_params(top=True,right=True,direction='in', which='both')


plt.savefig(f'plots/{key}_orbit.pdf')

plt.show()
