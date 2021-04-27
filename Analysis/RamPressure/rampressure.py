import pynbody
import pandas as pd
import numpy as np
import pickle
import sys
import tqdm
import os
from bulk import *
import fsps

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
    print(f'\t {sim}-{z0haloid}: Getting starting snapshot (dist = 2 Rvir)')
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
    print(f'\t {sim}-{z0haloid}: Start on snapshot {snap_start}, {filepaths[snap_start][-4:]}') # go down from there!
    return snap_start


def vec_to_xform(vec):
    vec_in = np.asarray(vec)
    vec_in = vec_in / np.sum(vec_in ** 2).sum() ** 0.5
    vec_p1 = np.cross([1, 0, 0], vec_in)
    vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
    vec_p2 = np.cross(vec_in, vec_p1)
    matr = np.concatenate((vec_p2, vec_in, vec_p1)).reshape((3, 3))
    return matr


def calc_ram_pressure(sim, z0haloid, filepaths, haloids, h1ids):
    output_tot = pd.DataFrame()
    
    #print('Starting calculations...')
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
        #print(f'\n\t Time = {t:.1f} Gyr')

        # RAM PRESSURE CALCULATIONS (SIMPLE)
        
        ## positions and velocities
        r_sat = np.array([sat.properties[k]/hubble*a for k in ['Xc','Yc','Zc']])
        r_host = np.array([host.properties[k]/hubble*a for k in ['Xc','Yc','Zc']])
        r_rel = r_sat - r_host
        h1dist = np.linalg.norm(r_rel)
        output['h1dist'] = [h1dist]
        print(f'\n\t {sim}-{z0haloid}: Distance from host = {h1dist:.2f} kpc')
        
        v_sat = np.array([sat.properties[k] for k in ['VXc','VYc','VZc']])
        v_host = np.array([host.properties[k] for k in ['VXc','VYc','VZc']])
        v_rel = v_sat - v_host
        v_rel_mag = np.linalg.norm(v_rel)

        print(f'\t {sim}-{z0haloid}: Relative velocity = {v_rel_mag:.2f} km/s')

        
        # nearest neighbor distance
        Xcs, Ycs, Zcs = np.array([]), np.array([]), np.array([])
        for halo in h:
            if len(halo.dm) > 1000:
                r = np.array([halo.properties[k]/hubble*a for k in ['Xc','Yc','Zc']])
                if not (r==r_sat).all():
                    Xcs = np.append(Xcs, r[0])
                    Ycs = np.append(Ycs, r[1])
                    Zcs = np.append(Zcs, r[2])
        
        
        r_others = np.array([Xcs, Ycs, Zcs]).T
        dists = np.linalg.norm(r_others - r_sat, axis=1)
        print(f'\t {sim}-{z0haloid}: dNN = {np.min(dists):.1f} kpc')
        
        ## galaxy properties
        M_star = np.sum(sat.s['mass'].in_units('Msol'))
        M_gas = np.sum(sat.g['mass'].in_units('Msol'))
        rvir = sat.properties['Rvir']/hubble*a
        h1rvir = host.properties['Rvir']/hubble*a

        output['M_star'] = [M_star]
        output['M_gas'] = [M_gas]
        output['satRvir'] = [rvir]
        output['hostRvir'] = [h1rvir]

        print(f'\t {sim}-{z0haloid}: Satellite M_gas = {M_gas:.1e} Msun')


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
        print(f'\t {sim}-{z0haloid}: Simple v_rel = {v_rel_mag:.1f}')
        print(f'\t {sim}-{z0haloid}: Simple rho_CGM = {rho_CGM:.1e}')
        print(f'\t {sim}-{z0haloid}: Simple P_ram = {Pram:.1e}')


        # RAM PRESSURE CALCULATIONS (ADVANCED)
        # roughly following the method of Simons et al: using a cylinder in front of the satellite
        # but we want to exclude those that have already been in the satellite
        
        # below code adapted from pynbody.analysis.angmom.sideon() to transform the snapshot so that the vector 'vel' points in the +y direction
        top = s
        print(f'\t {sim}-{z0haloid}: Centering positions')
        cen = pynbody.analysis.halo.center(sat, retcen=True)
        tx = pynbody.transformation.inverse_translate(top, cen)
        print(f'\t {sim}-{z0haloid}: Centering velocities')
        vcen = pynbody.analysis.halo.vel_center(sat, retcen=True) 
        tx = pynbody.transformation.inverse_v_translate(tx, vcen)
        
        print(f'\t {sim}-{z0haloid}: Getting velocity vector') 
        try:
            vel = np.average(sat.g['vel'], axis=0, weights=sat.g['mass'])
        except ZeroDivisionError:
            vel = np.average(sat.s['vel'], axis=0, weights=sat.s['mass'])
            
        vel_host = np.average(host.g['vel'], axis=0, weights=host.g['mass']) # not sure why I'm subtracting the host velocity but leaving it for now
        vel -= vel_host

        print(f'\t {sim}-{z0haloid}: Transforming snapshot')
        trans = vec_to_xform(vel)
        tx = pynbody.transformation.transform(tx, trans)
        
#         # get R_gal from the particletracking code
#         data = read_tracked_particles(sim, z0haloid)
#         tmin = np.unique(data.time)[np.argmin(np.abs(np.unique(data.time)-t))]
#         R_gal = np.mean(data[data.time==tmin].r_gal)
#         print(f'\t {sim}-{z0haloid}:  R_gal = {R_gal:.2f} kpc')
        
        
        radius = 0.5*rvir
        height = 0.75 * radius
        center = (0, rvir + height/2, 0)
        wind_filt = pynbody.filt.Disc(radius, height, cen=center)
        env = s[wind_filt].g
        print(f'\t {sim}-{z0haloid}: Identified {len(env)} gas particles to calculate wind properties')
        output['n_CGM'] = [len(env)]

        try:
            vel_CGM = np.linalg.norm(np.average(env['vel'],axis=0,weights=env['mass'])) # should be in units of Msun kpc**-3
            rho_CGM = np.average(env['rho'], weights=env['mass']) # should be in units of 
            std_rho_CGM = np.std(env['rho'])
            std_vel_CGM = np.std(np.linalg.norm(env['vel'], axis=1))
        except ZeroDivisionError:
            vel_CGM, rho_CGM = 0, 0
            std_vel_CGM, std_rho_CGM = 0, 0
            
        Pram = rho_CGM * vel_CGM * vel_CGM # overall units should be Msun kpc**-3 km s**-1
        
        output['vel_CGM_adv'] = [vel_CGM]
        output['rho_CGM_adv'] = [rho_CGM]
        output['std_vel_CGM'] = [std_vel_CGM]
        output['std_rho_CGM'] = [std_rho_CGM]
        output['Pram_adv'] = [Pram]

        print(f'\t {sim}-{z0haloid}: Advanced vel_CGM = {vel_CGM:.2f}')
        print(f'\t {sim}-{z0haloid}: Advanced rho_CGM = {rho_CGM:.1e}')
        print(f'\t {sim}-{z0haloid}: Advanced P_ram = {Pram:.1e}')
        
        #print(f'Mean vel of sat gas particles in this transform:', np.linalg.norm(np.mean(sat.g['vel'], axis=0, weights=sat.g['mass'])))

        # RESTORING PRESSURE CALCULATIONS
        try:
            pynbody.analysis.halo.center(sat)
            calc_rest = True
        except: 
            calc_rest = False
            Prest = 0.
            SigmaGas = 0.
            dphidz = 0.
        
        if calc_rest:
            p = pynbody.analysis.profile.Profile(s.g, min=0.01, max=rvir, ndim=3)
            percent_enc = p['mass_enc']/M_gas
            rhalf = np.min(p['rbins'][percent_enc > 0.5])
            SigmaGas = M_gas / (2*np.pi*rhalf**2)
            Rmax = sat.properties['Rmax']/hubble*a
            Vmax = sat.properties['Vmax']
            dphidz = Vmax**2 / Rmax
            Prest = dphidz * SigmaGas
        
        print(f'\t {sim}-{z0haloid}:  Prest = {Prest:.1e}')

        output['Prest'] = [Prest]
        output['SigmaGas'] = [SigmaGas]
        output['dphidz'] = [dphidz]
    
        
        # sfr calculations
        star_masses = np.array(sat.s['mass'].in_units('Msol'),dtype=float)
        star_metals = np.array(sat.s['metals'], dtype=float)
        star_ages = np.array(sat.s['age'].in_units('Myr'),dtype=float)
        size = len(star_ages)
        
        fsps_ssp = fsps.StellarPopulation(sfh=0,zcontinuous=1, imf_type=2, zred=0.0, add_dust_emission=False) # CHECK THESE
        solar_Z = 0.0196
        
        star_masses = star_masses[star_ages <= 100]
        star_metals = star_metals[star_ages <= 100]
        star_ages = star_ages[star_ages <= 100]
        print(f'\t {sim}-{z0haloid}: performing FSPS calculations on {len(star_masses)} star particles (subset of {size} stars)')
        if len(star_masses)==0:
            SFR = 0
        else:
            massform = np.array([])
            for age, metallicity, mass in zip(star_ages, star_metals, star_masses):
                fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
                mass_remaining = fsps_ssp.stellar_mass
                massform = np.append(massform, mass / np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining))

            SFR = np.sum(massform)/100e6
            
        output['SFR'] = [SFR]
        output['sSFR'] = [SFR/M_star]
        print(f'\t {sim}-{z0haloid}: sSFR = {SFR/M_star:.2e} yr**-1')
        
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
        h1ids = np.flip(h1ids[:snap_start+1])

    output_tot = calc_ram_pressure(sim, z0haloid, filepaths, haloids, h1ids)
    output_tot.to_hdf('../../Data/ram_pressure.hdf5',key=f'{sim}_{z0haloid}')







    





                
