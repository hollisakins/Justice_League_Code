# This code generates a figure showing the motion of a galaxy through the CGM alongside its ram pressure ratio over time
from analysis import *

sim = 'h229'
haloid = 22

def vec_to_xform(vec):
    vec_in = np.asarray(vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross([1, 0, 0], vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)
    matr = np.concatenate((vec_p2, vec_in, vec_p1)).reshape((3, 3))
    return matr

fig = plt.figure(figsize=(6.5, 4.5), constrained_layout=False)
# fig, ax = plt.subplots(1,1,figsize=(6.5, 4))

gs = mpl.gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.8], hspace=0.04, wspace=0.05, figure=fig)
gs.update(right=0.98, left=0.08, bottom=0.08, top=0.98)

ax = plt.subplot(gs[1,:])
img0 = plt.subplot(gs[0,0])
img1 = plt.subplot(gs[0,1])
img2 = plt.subplot(gs[0,2])
img3 = plt.subplot(gs[0,3])
img_axes = [img0,img1,img2,img3]

print('Plotting ram pressure')
data = read_ram_pressure(sim, haloid)

x = np.array(data.t,dtype=float)
y = np.array(data.Pram,dtype=float)/np.array(data.Prest,dtype=float)

ax.plot(x,y, label='Spherically-averaged CGM', color='k', linestyle='--')

x = np.array(data.t,dtype=float)
y = np.array(data.Pram_adv,dtype=float)/np.array(data.Prest,dtype=float)
ax.plot(x,y, label='True CGM', color='k', linestyle='-')
ax.legend(loc='upper left', frameon=False)

ax.semilogy()
ax.set_xlabel('Time [Gyr]')
ax.set_ylabel(r'$\mathcal{P} \equiv P_{\rm ram}/P_{\rm rest}$')

for i in img_axes:
    i.tick_params(labelleft=False, labelbottom=False)
    
img0.tick_params(left=True, labelleft=True)
img0.set_ylabel(r'$y$ [kpc]')

    
t0 = 4.534719
t1 = 7.120602
t2 = 8.608831
t3 = 13.262072

y0 = y[np.argmin(np.abs(x-t0))]
y1 = y[np.argmin(np.abs(x-t1))]
y2 = y[np.argmin(np.abs(x-t2))]
y3 = y[np.argmin(np.abs(x-t3))]

ax.set_xlim(3, 13.5)
ax.set_ylim(2e-4, 2e2)


ax.fill_between([3, t0, 5.5], [2e2, y0, 2e2], [2e2]*3, fc='0.95', ec='0.87')
ax.scatter([t0], [y0], fc='0.95', ec='0.87')

ax.fill_between([5.65, t1, 8.15], [2e2, y1, 2e2], [2e2]*3, fc='0.95', ec='0.9')
ax.scatter([t1], [y1], fc='0.95', ec='0.87')

ax.fill_between([8.3, t2, 10.8], [2e2, y2, 2e2], [2e2]*3, fc='0.95', ec='0.9')
ax.scatter([t2], [y2], fc='0.95', ec='0.87')

ax.fill_between([11, t3, 13.5], [2e2, y3, 2e2], [2e2]*3, fc='0.95', ec='0.9')
ax.scatter([t3], [y3], fc='0.95', ec='0.87')

ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))


print('Getting filepaths for four snapshots...')
filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,haloid)
snap_start = get_snap_start(sim,haloid)
    
if len(haloids) >= snap_start:
    filepaths = np.flip(filepaths[:snap_start+1])
    haloids = np.flip(haloids[:snap_start+1])
    h1ids = np.flip(h1ids[:snap_start+1])

ts = np.array([t0,t1,t2,t3])
ys = np.array([y0,y1,y2,y3])
fs, hs, h1s = np.array([]),np.array([]),np.array([])
for i, filepath in enumerate(filepaths):
    s = pynbody.load(filepath)
    h = haloids[i]
    h1 = h1ids[i]
    t = s.properties['time'].in_units('Gyr')
    if any((t - ts) < 0.05):
        fs = np.append(fs, filepath)
        hs = np.append(hs, h)
        h1s = np.append(h1s, h1)

i = 1
for iax,t,y,f,hid,h1id in zip(img_axes,ts,ys,fs,hs,h1s):
    print(f'Loading snap {i}')
    i += 1
    
    s = pynbody.load(f)
    s.physical_units()
    h = s.halos()
    halo = h[hid]
    host = h[h1id] # may not always be halo 1! (but probably is)
    a = s.properties['a']
    print('\t Made halo catalog')
        
    # below code adapted from pynbody.analysis.angmom.sideon()
    top = s
    print('\t Centering positions')
    cen = pynbody.analysis.halo.center(halo, retcen=True)
    tx = pynbody.transformation.inverse_translate(top, cen)
    print('\t Centering velocities')
    vcen = pynbody.analysis.halo.vel_center(halo, retcen=True) 
    tx = pynbody.transformation.inverse_v_translate(tx, vcen)

    print('\t Getting velocity vector') # may want to get only from inner 10 kpc
    try:
        vel = np.average(halo.g['vel'], axis=0, weights=halo.g['mass'])
    except ZeroDivisionError:
        vel = np.average(halo.s['vel'], axis=0, weights=halo.s['mass'])

    vel_host = np.average(host.g['vel'], axis=0, weights=host.g['mass']) 
    vel -= vel_host
    
    
#     gvel = halo.g['vel']
#     gr = np.array(halo.g['r'].in_units('kpc'),dtype=float)
#     gmass = halo.g['mass']
#     vel = np.average(gvel[gr < 10], axis=0, weights=gmass[gr < 10])
#     Rvir = halo.properties['Rvir']/hubble*a
#     sphere1 = pynbody.filt.Sphere(f'{round(Rvir,0)} kpc')
#     sphere2 = pynbody.filt.Sphere(f'{round(1.5*Rvir,0)} kpc')
#     ssub = s.g[sphere2 & ~sphere1]
#     vel_CGM = np.average(ssub['vel'], axis=0, weights=ssub['mass'])
#     vel -= vel_CGM
    
    print('\t Transforming snapshot')
    trans = vec_to_xform(vel)
    tx = pynbody.transformation.transform(tx, trans)
    
    smin, smax = -30, 30
    gas_vmin, gas_vmax = 6e2, 3e5
    
    print('\t Making gas image')    
    im = pynbody.plot.sph.velocity_image(s.g[pynbody.filt.Sphere('%s kpc' % str(1.5*(smax-smin)))], width='%s kpc' % str(smax-smin),
                                         cmap='viridis', vmin=gas_vmin, vmax=gas_vmax,
                                         vector_color='cyan', vector_resolution = 15, av_z='rho', ret_im=True, denoise=False,
                                         approximate_fast=False, subplot=iax, show_cbar=False, quiverkey=False)

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    size = 20
    if i==1:
        bar = AnchoredSizeBar(iax.transData, size, str(size)+' kpc', loc='lower right', bbox_to_anchor=(1.,1.),
                              bbox_transform=iax.transAxes, color='k', frameon=False)
        iax.add_artist(bar)

        
fig.savefig('plots/ram_pressure_image.pdf',dpi=300)
plt.close()