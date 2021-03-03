import numpy as np
import pynbody
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from bulk import *

mpl.rc('font',**{'family':'serif','monospace':['Palatino']})
mpl.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = False
mpl.rcParams.update({'font.size': 9})

age = 13.800797497330507
hubble =  0.6776942783267969

sim = 'h242'
haloid = 80
filepaths, haloids, h1ids = get_filepaths_haloids(sim,haloid)
path = filepaths[0]
print('Loading',path)

s = pynbody.load(path)
s.physical_units()
h = s.halos()

print('Centering halo 1')
pynbody.analysis.angmom.center(h[1])


fig = plt.figure(dpi=300, figsize=(7,3), constrained_layout=True)
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios = [1,1.2], figure=fig)
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax = [ax0,ax1]

a = float(s.properties['a'])
host_Rvir = h[1].properties['Rvir'] / hubble * a
width = round(2.8*host_Rvir, 1)

print('Making gas image')    
im = pynbody.plot.sph.velocity_image(s.g, 
                                     width=f'{width:.1f} kpc', # set to 2.8 Rvir
                                     cmap='viridis'
                                     vector_color = 'cyan', 
                                     vector_resolution = 15, 
                                     av_z = False, # don't average: we want a slice 
                                     ret_im=True, denoise=False, approximate_fast=False, subplot=ax[0], show_cbar=False, quiverkey=False)

print('Plotting circle')
circ = plt.Circle((0,0), host_Rvir, color = '0.7', linestyle='-', fill=False, linewidth=1)
ax[0].add_artist(circ)

print('Plotting satellite orbit')
data = read_tracked_particles(sim, haloid)
X = np.array([])
Y = np.array([])
for t in np.unique(data.time):
    d = data[data.time==t]
    x = np.mean(d.sat_Xc) - np.mean(d.host_Xc)
    y = np.mean(d.sat_Yc) - np.mean(d.host_Yc)
    X = np.append(X,x)
    Y = np.appedn(Y,y)
    
ax[0].plot(X,Y, color='w', linestyle='--')

ax[0].set_xlabel(r'$x$ [kpc]')
ax[0].set_ylabel(r'$y$ [kpc]')

print('Plotting ram pressure')
data = pd.read_hdf('../../ram_pressure.hdf5', key=f'{sim}_{str(haloid)}')
x = np.array(data.t,dtype=float)
y = np.array(data.Pram,dtype=float)/np.array(data.Prest,dtype=float)
ax[1].plot(x,y, label='Spherically-averaged CGM', color='k', linestyle='--')

x = np.array(data.t,dtype=float)
y = np.array(data.Pram_adv,dtype=float)/np.array(data.Prest,dtype=float)
ax[1].plot(x,y, label='True CGM', color='k', linestyle='-')
ax[1].legend(loc='upper left', frameon=False)

ax[1].semilogy()
ax[1].set_xlabel('Time [Gyr]')
ax[1].set_ylabel(r'$\mathcal{P} \equiv P_{\rm ram}/P_{\rm rest}$')

plt.savefig('figures/ram_pressure.pdf')
plt.close()
    