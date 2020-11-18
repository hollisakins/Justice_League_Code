import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys

mpl.rc('font',**{'family':'serif','monospace':['Palatino']})
mpl.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = False
mpl.rcParams.update({'font.size': 9})

age = 13.800797497330507


def read_timescales():
    '''Function to read in the data file which contains quenching and infall times'''
    data = []
    with open('../../Data/QuenchingTimescales.data', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f,encoding='latin1'))
            except EOFError:
                break

    data = pd.DataFrame(data)
    return data


# key = 'h242_24'
key = str(sys.argv[1])
path = '../../Data/tracked_particles.hdf5'
data = pd.read_hdf(path, key=key)
data

# Temporary code to fix classification issue 
# (forgot to exclude disk particles from the halo in particletracking.py, so some were being classified in both)

data = data.rename(columns={'sat_disk': 'sat_disk_wrong',
                            'host_disk': 'host_disk_wrong',
                            'sat_halo': 'sat_halo_wrong',
                            'host_halo': 'host_halo_wrong'})

data['sat_disk'] = (data.rho >= 0.1) & (data.temp <= 1.2e4) & (data.r <= 3)
data['sat_halo'] = (data.r_per_Rvir < 1) & ~data.sat_disk
data['host_disk'] = (data.rho >= 0.1) & (data.temp <= 1.2e4) & (data.r_per_Rvir > 1) & (data.h1dist < 0.1)
data['host_halo'] = (data.r_per_Rvir > 1) & (data.h1dist < 1) & ~data.host_disk

data


times = np.unique(data.time)

frac_satdisk = np.array([])
frac_sathalo = np.array([])
frac_hostdisk = np.array([])
frac_hosthalo = np.array([])
frac_IGM = np.array([])

for t in times:
    d = data[data.time==t]
    
    frac_satdisk = np.append(frac_satdisk,np.sum(d.mass[d.sat_disk])/np.sum(d.mass))
    frac_sathalo = np.append(frac_sathalo,np.sum(d.mass[d.sat_halo])/np.sum(d.mass))
    frac_hostdisk = np.append(frac_hostdisk,np.sum(d.mass[d.host_disk])/np.sum(d.mass))
    frac_hosthalo = np.append(frac_hosthalo,np.sum(d.mass[d.host_halo])/np.sum(d.mass))
    frac_IGM = np.append(frac_IGM,np.sum(d.mass[d.IGM])/np.sum(d.mass))

frac_lost = 1 - (frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_IGM)

timescales = read_timescales()
timescales = timescales[(timescales.sim==key[:4])&(timescales.haloid==int(key[-2:]))]
tinfall = age - timescales.tinfall.tolist()[0]
tquench = age - timescales.tquench.tolist()[0]


fig, ax = plt.subplots(1,1,dpi=300,figsize=(4,3))

colors = ['mediumblue', 'tab:red', 'darkorchid', 'darkorange', 'k']

lw = 0.6
alpha = 0.3
fontsize = 8

ax.fill_between(times, 0, frac_satdisk,fc=colors[0], alpha=alpha)
ax.plot(times, frac_satdisk, color=colors[0], linewidth=lw, zorder=6)

ax.fill_between(times, frac_satdisk, frac_satdisk+frac_sathalo, fc=colors[1], alpha=alpha)
ax.plot(times, frac_satdisk+frac_sathalo, color=colors[1], linewidth=lw, zorder=5)

ax.fill_between(times, frac_satdisk+frac_sathalo, frac_satdisk+frac_sathalo+frac_hostdisk, fc=colors[2], alpha=alpha)
ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk, color=colors[2], linewidth=lw, zorder=4)

ax.fill_between(times, frac_satdisk+frac_sathalo+frac_hostdisk, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo, fc=colors[3], alpha=alpha)
ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo, color=colors[3], linewidth=lw, zorder=3)

ax.fill_between(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo,frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_IGM, fc=colors[4], alpha=alpha)
ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_IGM, color=colors[4], linewidth=lw, zorder=2)

ax.axline((tinfall,0),(tinfall,1),linestyle='--', linewidth=0.5, color='k')
ax.axline((tquench,0),(tquench,1),linestyle=':', linewidth=0.5, color='k')

ax.set_xlim(min(times),max(times))
ax.set_ylim(0,1)

ax.set_xlabel('Time [Gyr]')
ax.set_ylabel(r'$f(M_{\rm gas})$')

# ax.annotate('Satellite \n Disk',(7.3,0.08),ha='center', va='center', color=colors[0], size=fontsize)
# ax.annotate('Satellite Halo',(6.7,0.5),ha='center', va='center', color=colors[1], size=fontsize)
# ax.annotate('Host Disk',(12.5,0.13),ha='center', va='center', color=colors[2], size=fontsize)
# ax.annotate('Host Halo',(12.2,0.55),ha='center', va='center', color=colors[3], size=fontsize)
# ax.annotate('IGM',(9.2,0.95),ha='center', va='center', color=colors[4], size=fontsize)

ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
ax.tick_params(direction='in', which='both', top=True,right=True)

ax.annotate(key.replace('_','-'), (0.94, 0.92), xycoords='axes fraction', ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='w', ec='0.5', alpha=0.9), zorder=100)

plt.savefig(f'plots/fractions/{key}_fractions.pdf')
plt.close()