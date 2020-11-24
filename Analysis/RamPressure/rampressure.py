import pynbody
import pandas as pd
import numpy as np
import pickle
import sys
import tqdm
import os

hubble =  0.6776942783267969

# h148 28
# h148 37
# h148 68
# h242 80
# h229 20
# h229 22
# h242 24
# h148 13


def get_stored_filepaths_haloids(sim,z0haloid):
    # get snapshot paths and haloids from stored file
    with open('../../Data/filepaths_haloids.pickle','rb') as f:
        d = pickle.load(f)
    try:
        filepaths = d['filepaths'][sim]
    except KeyError:
        print("sim must be one of 'h148','h229','h242','h329'")
        raise
    try:
        haloids = d['haloids'][sim][z0haloid]
        h1ids = d['haloids'][sim][1]
    except KeyError:
        print('z0haloid not found, perhaps this is a halo that has no stars at z=0, and therefore isnt tracked')
        raise
    return filepaths,haloids,h1ids
    

def read_timesteps(simname):
    data = []
    with open(f'../../Data/timesteps_data/{simname}.data','rb') as f:
        while True:
            try: 
                data.append(pickle.load(f))
            except EOFError:
                break
    
    data = pd.DataFrame(data)
    return data

def get_snap_start(sim,z0haloid):
    print('Getting starting snapshot (dist = 2 Rvir)')
    filepaths,haloids,h1ids = get_stored_filepaths_haloids(sim,z0haloid)
    ts = read_timesteps(sim)
    ts = ts[ts.z0haloid == z0haloid]

    dist = np.array(ts.h1dist, dtype=float)
    time = np.array(ts.time, dtype=float)
    ti = np.min(time[dist <= 2])

    for i,f in enumerate(filepaths):
        s = pynbody.load(f)
        t = float(s.properties['time'].in_units('Gyr'))
        if t<ti:
            snap_start = i
            break
        else: 
            continue
    print(f'Start on snapshot {snap_start}, {filepaths[snap_start][-4:]}') # go down from there!
    return snap_start


def calc_ram_pressure(sim, z0haloid, filepaths, haloids, h1ids):
    output_tot = pd.DataFrame()
    
    print('Starting calculations...')
    for f,haloid,h1id in tqdm.tqdm(zip(filepaths,haloids,h1ids),total=len(filepaths)):
        s = pynbody.load(f)
        s.physical_units()
        h = s.halos()
        sat = h[haloid]
        host = h[h1id]
        t = float(s.properties['time'].in_units('Gyr'))
        a = float(s.properties['a'])

        output = pd.DataFrame()
        output['t'] = [t]
        output['a'] = [a]
        print(f'\n\t Time = {t:.1f} Gyr')

        # RAM PRESSURE CALCULATIONS (SIMPLE)

        ## positions and velocities
        r_sat = np.array([sat.properties[k]/hubble for k in ['Xc','Yc','Zc']])
        r_host = np.array([host.properties[k]/hubble for k in ['Xc','Yc','Zc']])
        r_rel = r_sat - r_host
        h1dist = np.linalg.norm(r_rel)
        output['h1dist'] = [h1dist]
        print(f'\t Distance from host = {h1dist:.2f} kpc')
        
        v_sat = np.array([sat.properties[k] for k in ['VXc','VYc','VZc']])
        v_host = np.array([host.properties[k] for k in ['VXc','VYc','VZc']])
        v_rel = v_sat - v_host
        v_rel_mag = np.linalg.norm(v_rel)

        print(f'\t Relative velocity = {v_rel_mag:.2f} km/s')

        ## galaxy properties
        M_star = np.sum(sat.s['mass'].in_units('Msol'))
        M_gas = np.sum(sat.g['mass'].in_units('Msol'))
        rvir = sat.properties['Rvir']/hubble
        h1rvir = host.properties['Rvir']/hubble

        output['Rvir'] = [rvir]
        output['M_star'] = [M_star]
        output['M_gas'] = [M_gas]
        output['hostRvir'] = [h1rvir]

        print(f'\t Satellite M_gas = {M_gas:.1e} Msun')


        ## calculate rho_CGM from spherical density profile
        pynbody.analysis.halo.center(host)
        pg = pynbody.analysis.profile.Profile(s.g, min=0.01, max=2*h1dist, ndim=3)
        rbins = pg['rbins']
        density = pg['density']

        rho_CGM = density[np.argmin(np.abs(rbins-h1dist))]
        Pram = rho_CGM * v_rel_mag * v_rel_mag
        output['vel_CGM'] = [v_rel_mag]
        output['rho_CGM'] = [rho_CGM]
        output['Pram'] = [Pram]
        print(f'\t Simple rho_CGM = {rho_CGM:.1e}')
        print(f'\t Simple P_ram = {Pram:.1e}')


        # RAM PRESSURE CALCULATIONS (ADVANCED)

        # we want to include all gas particles with R/Rvir between 1 and 2 (just outside the satellite)
        # but we want to exclude those that have already been in the satellite

        # center the positions and velocities of the satellite
        pynbody.analysis.halo.center(sat)

        r_inner = rvir
        r_outer = 2*rvir
        inner_sphere = pynbody.filt.Sphere(str(r_inner)+' kpc', [0,0,0])
        outer_sphere = pynbody.filt.Sphere(str(r_outer)+' kpc', [0,0,0])

        env = s[outer_sphere & ~inner_sphere].gas
        print(f'\t Identified {len(env)} gas particles to calculate wind properties')

        key = str(sim)+'_'+str(z0haloid)
        path = '../../Data/tracked_particles.hdf5'
        data = pd.read_hdf(path, key=key)
        data = data[(data.r_per_Rvir < 1)&(data.time < t)]
        iords_to_exclude = np.array(data.pid,dtype=int)

        exclude = np.isin(env['iord'], iords_to_exclude)
        env = env[~exclude]
        print(f'\t Reduced to {len(env)} particles by excluding those that were prev. in sat')

        # since satellite is centered, this is vel relative to satellite 
        vel_CGM = np.linalg.norm(np.average(env['vel'],axis=0,weights=env['mass']))
        rho_CGM = np.average(env['rho'], weights=env['mass'])
        Pram = rho_CGM * vel_CGM * vel_CGM
        output['vel_CGM_adv'] = [vel_CGM]
        output['rho_CGM_adv'] = [rho_CGM]
        output['Pram_adv'] = [Pram]

        print(f'\t Advanced rho_CGM = {rho_CGM:.1e}')
        print(f'\t Advanced P_ram = {Pram:.1e}')


        # DIFFERENTIAL TIDAL FORCE CALCULATIONS
        # it would not be simple to calculate the differential tidal for DeltaF, since that would be the force on a small 
        # bit of mass dm
        # instead we will calculated the Roche Limit, the radius at which the satellite would be tidally disrupted

        # unsure what delta_r to choose. for now using Rvir
        Delta_r = 0.2*rvir
        sat_sphere = pynbody.filt.Sphere(str(Delta_r)+' kpc', [0,0,0])
        m_sat_enc = np.sum(sat[sat_sphere]['mass'].in_units('Msol'))
        host_sphere = pynbody.filt.Sphere(str(h1dist)+' kpc', [0,0,0])
        M_host_enc = np.sum(host[host_sphere]['mass'].in_units('Msol'))

        r_R = (2*M_host_enc / m_sat_enc)**(1/3) * Delta_r
        output['M_host_enc'] = [M_host_enc]
        output['m_sat_enc'] = [m_sat_enc]
        output['r_R'] = [r_R]

        print(f'\t Roche limit r_R = {r_R:.2f} kpc')
        print(f'\t Host R_vir = {h1rvir:.2f} kpc')


        # RESTORING PRESSURE CALCULATIONS
        try:
            pynbody.analysis.halo.center(sat)
        except: 
            Prest = 0

        if Prest != 0:
            p = pynbody.analysis.profile.Profile(s.g, min=0.01, max=rvir, ndim=3)
            percent_enc = p['mass_enc']/M_gas
            rhalf = np.min(p['rbins'][percent_enc > 0.5])
            SigmaGas = M_gas / (2*np.pi*rhalf**2)
            Rmax = sat.properties['Rmax']
            Vmax = sat.properties['Vmax']
            dphidz = Vmax**2 / Rmax
            Prest = dphidz * SigmaGas
        
        print(f'\t Prest = {Prest:.1e}')

        output['Prest'] = [Prest]


        output_tot = pd.concat([output_tot, output])

    return output_tot


if __name__ == '__main__':
    sim = str(sys.argv[1])
    z0haloid = int(sys.argv[2])
    
    snap_start = get_snap_start(sim,z0haloid)
    filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,z0haloid)

    # fix the case where the satellite doesn't have merger info prior to 
    if len(haloids) < snap_start:
        snap_start = len(haloids)
        raise Exception('Careful! You may have an error since the satellite doesnt have mergertree info out to the time where you want to start. This case is untested')
    
    if len(haloids) >= snap_start:
        filepaths = np.flip(filepaths[:snap_start+1])
        haloids = np.flip(haloids[:snap_start+1])

    output_tot = calc_ram_pressure(sim, z0haloid, filepaths, haloids, h1ids)
    output_tot.to_hdf('../../Data/ram_pressure.hdf5',key=f'{sim}_{z0haloid}')







    





                
