import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
from bulk import *

mpl.rc('font',**{'family':'serif','monospace':['Palatino']})
mpl.rc('text', usetex=True)
plt.rcParams['figure.constrained_layout.use'] = False
mpl.rcParams.update({'font.size': 9})

age = 13.800797497330507


def read_timesteps(sim):
    '''Function to read in the data file which contains quenching and infall times'''
    data = []
    with open(f'../../Data/timesteps_data/{sim}.data', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f,encoding='latin1'))
            except EOFError:
                break

    data = pd.DataFrame(data)
    return data

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

def read_infall_properties():
    '''Function to read in the data file with quenching timescales and satellite properties at infall.'''
    data = []
    with open(f'../../Data/QuenchingTimescales_InfallProperties.data','rb') as f:
        while True:
            try: 
                data.append(pickle.load(f))
            except EOFError:
                break
            
    data = pd.DataFrame(data)
    data['timescale'] = data.tinfall - data.tquench
    
    return data


def fill_fractions_ax(key, ax, label=False, show_y_ticks=False):
    sim = str(key[:4])
    haloid = int(key[5:])
    print(sim,haloid)
    data = read_tracked_particles(sim,haloid,verbose=False)
    
    times = np.unique(data.time)

    frac_satdisk = np.array([])
    frac_sathalo = np.array([])
    frac_hostdisk = np.array([])
    frac_hosthalo = np.array([])
    frac_othersat = np.array([])
    frac_IGM = np.array([])

    for t in times:
        d = data[data.time==t]

        frac_satdisk = np.append(frac_satdisk,np.sum(d.mass[d.sat_disk])/np.sum(d.mass))
        frac_sathalo = np.append(frac_sathalo,np.sum(d.mass[d.sat_halo])/np.sum(d.mass))
        frac_hostdisk = np.append(frac_hostdisk,np.sum(d.mass[d.host_disk])/np.sum(d.mass))
        frac_hosthalo = np.append(frac_hosthalo,np.sum(d.mass[d.host_halo])/np.sum(d.mass))
        frac_othersat = np.append(frac_othersat,np.sum(d.mass[d.other_sat])/np.sum(d.mass))
        frac_IGM = np.append(frac_IGM,np.sum(d.mass[d.IGM])/np.sum(d.mass))

    frac_lost = 1 - (frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_othersat+frac_IGM)

    timescales = read_timescales()
    timescales = timescales[(timescales.sim==key[:4])&(timescales.haloid==int(key[5:]))]
    tinfall = age - timescales.tinfall.tolist()[0]
    tquench = age - timescales.tquench.tolist()[0]
    
    times = np.unique(data.time)

    frac_satdisk,frac_sathalo,frac_hostdisk,frac_hosthalo,frac_othersat,frac_IGM = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

    for t in times:
        d = data[data.time==t]
        frac_satdisk = np.append(frac_satdisk,np.sum(d.mass[d.sat_disk])/np.sum(d.mass))
        frac_sathalo = np.append(frac_sathalo,np.sum(d.mass[d.sat_halo])/np.sum(d.mass))
        frac_hostdisk = np.append(frac_hostdisk,np.sum(d.mass[d.host_disk])/np.sum(d.mass))
        frac_hosthalo = np.append(frac_hosthalo,np.sum(d.mass[d.host_halo])/np.sum(d.mass))
        frac_othersat = np.append(frac_othersat,np.sum(d.mass[d.other_sat])/np.sum(d.mass))
        frac_IGM = np.append(frac_IGM,np.sum(d.mass[d.IGM])/np.sum(d.mass))

    frac_lost = 1 - (frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_othersat+frac_IGM)

    lw = 0.6
    alpha = 0.3
    fontsize = 8
    colors = ['mediumblue', 'tab:red', 'darkorchid', 'darkorange', 'g', 'k']

    ax.fill_between(times, 0, frac_satdisk,fc=colors[0], alpha=alpha)
    ax.plot(times, frac_satdisk, color=colors[0], linewidth=lw, zorder=6)

    ax.fill_between(times, frac_satdisk, frac_satdisk+frac_sathalo, fc=colors[1], alpha=alpha)
    ax.plot(times, frac_satdisk+frac_sathalo, color=colors[1], linewidth=lw, zorder=5)

    ax.fill_between(times, frac_satdisk+frac_sathalo, frac_satdisk+frac_sathalo+frac_hostdisk, fc=colors[2], alpha=alpha)
    ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk, color=colors[2], linewidth=lw, zorder=4)

    ax.fill_between(times, frac_satdisk+frac_sathalo+frac_hostdisk, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo, fc=colors[3], alpha=alpha)
    ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo, color=colors[3], linewidth=lw, zorder=3)

    ax.fill_between(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo,frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_othersat, fc=colors[4], alpha=alpha)
    ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_othersat, color=colors[4], linewidth=lw, zorder=2)
    
    ax.fill_between(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_othersat,frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_othersat+frac_IGM, fc=colors[5], alpha=alpha)
#     ax.plot(times, frac_satdisk+frac_sathalo+frac_hostdisk+frac_hosthalo+frac_IGM, color=colors[5], linewidth=lw, zorder=2)

    ax.axline((tinfall,0),(tinfall,1),linestyle='--', linewidth=0.5, color='k')
    ax.axline((tquench,0),(tquench,1),linestyle=':', linewidth=0.5, color='k')

    ax.set_xlim(min(times),max(times))
    ax.set_ylim(0,1)

    if label:
        if key=='h148-37':
            ax.annotate('Sat \n Disk',(8.2,0.8),ha='center', va='center', color=colors[0], size=fontsize)
            ax.annotate('Sat \n Halo',(8.2,0.65),ha='center', va='center', color=colors[1], size=fontsize)
            ax.annotate('Host Disk',(12,0.42),ha='center', va='center', color=colors[2], size=fontsize)
            ax.annotate('Host \n Halo',(11.2,0.65),ha='center', va='center', color=colors[3], size=fontsize)
            ax.annotate('IGM',(9.3,0.87),ha='left', va='center', color=colors[5], size=fontsize)
#         if key=='h148_68':
#             ax.annotate('Sat \n Disk',(7.3,0.1),ha='center', va='center', color=colors[0], size=fontsize)
#             ax.annotate('Sat \n Halo',(7.3,0.53),ha='center', va='center', color=colors[1], size=fontsize)
#             ax.annotate('Host \n Disk',(13,0.33),ha='center', va='center', color=colors[2], size=fontsize)
#             ax.annotate('Host Halo',(10.8,0.55),ha='center', va='center', color=colors[3], size=fontsize)
#             ax.annotate('IGM',(9.2,0.95),ha='center', va='center', color=colors[5], size=fontsize)
    if not show_y_ticks:
        ax.tick_params(labelleft=False)

    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.tick_params(direction='in', which='both', top=True,right=True)

    
    for sim, simname in {'h148':'Sandra', 'h229':'Ruth', 'h242':'Sonia', 'h329':'Elena'}.items():
        key = key.replace(sim, simname)
        
    key = key.replace('_','-')
    ax.annotate(key, (0.94, 0.92), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', fc='w', ec='0.5', alpha=0.9), zorder=100)

    

path = '../../Data/tracked_particles.hdf5'
with pd.HDFStore(path) as hdf:
    keys = [k[1:] for k in hdf.keys()]
    print('Available keys:', *keys)

timescales = read_timescales()
taus = np.array([])
for key in keys:
    sim = key[:4]
    haloid = int(key[5:])
    ts = timescales[(timescales.sim==sim)&(timescales.haloid==haloid)]
    taus = np.append(taus, ts.tinfall.iloc[0] - ts.tquench.iloc[0])
    
keys = np.array(keys)
keys = keys[~np.isnan(taus)]
taus = taus[~np.isnan(taus)]
print(*np.array(keys)[np.argsort(taus)])

###

keys = ['h329_33','h148_278','h229_27','h242_41','h148_13','h229_23','h229_55','h148_283','h148_45','h242_24','h229_22','h148_68','h148_37','h242_80','h229_20','h148_28']

fig = plt.figure(figsize=(7.5, 7.5), dpi=300)
gs = mpl.gridspec.GridSpec(nrows=4, ncols=4, figure=fig)
gs.update(wspace=0.05, top=0.98, right=0.99, left=0.07, bottom=0.07)

ax0,ax1,ax2,ax3 = plt.subplot(gs[0,0]),plt.subplot(gs[0,1]),plt.subplot(gs[0,2]),plt.subplot(gs[0,3])
ax4,ax5,ax6,ax7 = plt.subplot(gs[1,0]),plt.subplot(gs[1,1]),plt.subplot(gs[1,2]),plt.subplot(gs[1,3])
ax8,ax9,ax10,ax11 = plt.subplot(gs[2,0]),plt.subplot(gs[2,1]),plt.subplot(gs[2,2]),plt.subplot(gs[2,3])
ax12,ax13,ax14,ax15 = plt.subplot(gs[3,0]),plt.subplot(gs[3,1]),plt.subplot(gs[3,2]),plt.subplot(gs[3,3])

fill_fractions_ax(keys[0],ax0, show_y_ticks=True)
fill_fractions_ax(keys[1],ax1)
fill_fractions_ax(keys[2],ax2)
fill_fractions_ax(keys[3],ax3)
fill_fractions_ax(keys[4],ax4, show_y_ticks=True)
fill_fractions_ax(keys[5],ax5)
fill_fractions_ax(keys[6],ax6)
fill_fractions_ax(keys[7],ax7)
fill_fractions_ax(keys[8],ax8, show_y_ticks=True)
fill_fractions_ax(keys[9],ax9)
fill_fractions_ax(keys[10],ax10)
fill_fractions_ax(keys[11],ax11)
fill_fractions_ax(keys[12],ax12, label=True, show_y_ticks=True)
fill_fractions_ax(keys[13],ax13)
fill_fractions_ax(keys[14],ax14)
fill_fractions_ax(keys[15],ax15)

fig.text(0.53, 0.04, 'Time [Gyr]', ha='center', va='center')
fig.text(0.02, 0.53, r'$f(M_{\rm gas})$', ha='center', va='center', rotation='vertical')

plt.savefig(f'plots/fractions/fractions_big.pdf')
plt.show()