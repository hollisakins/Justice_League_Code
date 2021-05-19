 # The purpose of this file is to perform a series of data manipuation and processing commands to particle tracking data in bulk. 
# In particular, functions in this file import particle tracking and ram pressure data, join them as necessary, calculate kinetic 
# and potential energies of particles, classify particles as disk vs. halo, identify ejected or expulsed particles, and more. 
# The reason these functions are written here is so that we can ensure that we are using the same data processing procedures throughout 
# the analysis and not have to repeat this code for each analysis component. 
import pynbody
import pandas as pd
import numpy as np
import pickle
from base import *

def get_keys():
    path = '../../Data/ejected_particles.hdf5'
    with pd.HDFStore(path) as hdf:
        keys = [k[1:] for k in hdf.keys()]
    print(*keys)
    return keys


def read_tracked_particles(sim, haloid, verbose=False):
    
    if verbose: print(f'Loading tracked particles for {sim}-{haloid}...')
    
    key = f'{sim}_{str(int(haloid))}'

    # import the tracked particles dataset
    path = '../../Data/tracked_particles.hdf5'
    data = pd.read_hdf(path, key=key)
    
    time = np.unique(data.time)
    dt = time[1:]-time[:-1]
    dt = np.append(dt[0], dt)
    dt = dt[np.unique(data.time, return_inverse=True)[1]]
    data['dt'] = dt
    
    
    if verbose: print('Successfully loaded')
    
    r_gal = np.array([])
    for t in np.unique(data.time):
        d = data[data.time==t]
        r_gas = np.mean(d.sat_r_gas)
        r_half = np.mean(d.sat_r_half)
        rg = np.max([r_gas,r_half])

        if np.isnan(rg):
            rg = r_gal_prev

        if verbose: print(f't = {t:1f} Gyr, satellite R_gal = {rg:.2f} kpc')
        r_gal = np.append(r_gal,[rg]*len(d))

        r_gal_prev = rg

    data['r_gal'] = r_gal
    
    r_gal_prev = 0
    r_gal = np.array([])
    for t in np.unique(data.time):
        d = data[data.time==t]
        r_gas = np.mean(d.host_r_gas)
        r_half = np.mean(d.host_r_half)
        rg = np.max([r_gas,r_half])

        if np.isnan(rg):
            rg = r_gal_prev

        if verbose: print(f't = {t:1f} Gyr, host R_gal = {rg:.2f} kpc')
        r_gal = np.append(r_gal,[rg]*len(d))

        r_gal_prev = rg

    data['host_r_gal'] = r_gal
    
    thermo_disk = (np.array(data.temp) < 1.2e4) & (np.array(data.rho) > 0.1)
    
    in_sat = np.array(data.in_sat)
    other_sat = np.array(data.in_other_sat)
    in_host = np.array(data.in_host) & ~in_sat & ~other_sat
    
    sat_disk = in_sat & thermo_disk
    sat_halo = in_sat & ~thermo_disk
    
    host_disk = in_host & thermo_disk
    host_halo = in_host & ~thermo_disk
    
    IGM = np.array(data.in_IGM)
    
    
#    sat_disk = in_sat & (np.array(data.r) <= np.array(data.r_gal))
#     sat_halo = in_sat & (np.array(data.r) > np.array(data.r_gal))
#     sat_cool_disk = sat_disk & thermo_disk
#     sat_hot_disk = sat_disk & ~thermo_disk
#     sat_cool_halo = sat_halo & thermo_disk
#     sat_hot_halo = sat_halo & ~thermo_disk

#     in_host = np.array(data.in_host) & ~in_sat
#     host_disk = in_host & (np.array(data.r_rel_host) <= np.array(data.host_r_gal))
#     host_halo = in_host & (np.array(data.r_rel_host) > np.array(data.host_r_gal))

#     other_sat = np.array(data.in_other_sat)
#     IGM = np.array(data.in_IGM)
    
    
    # basic classifications
    data['sat_disk'] = sat_disk
    data['sat_halo'] = sat_halo
    data['host_disk'] = host_disk
    data['host_halo'] = host_halo
    data['other_sat'] = other_sat
    data['IGM'] = IGM
    
    # more advanced classifications
    #data['cool_disk'] = sat_cool_disk
    #data['hot_disk'] = sat_hot_disk
    #data['cool_halo'] = sat_cool_halo
    #data['hot_halo'] = sat_hot_halo

    return data

def calc_angles(d):
    # get gas particle velocity
    v = np.array([d.vx,d.vy,d.vz])

    # get velocity of CGM wind (host velocity relative to satellite)
    v_sat = np.array([d.sat_vx,d.sat_vy,d.sat_vz])
    v_host = np.array([d.host_vx,d.host_vy,d.host_vz])
    v_rel = v_host - v_sat # we want the velocity of the host in the satellite rest frame

    # take the dot product and get the angle, in degrees
    v_hat = v / np.linalg.norm(v)
    v_rel_hat = v_rel / np.linalg.norm(v_rel)
    angle = np.arccos(np.dot(v_hat,v_rel_hat)) * 180/np.pi

    d['angle'] = angle
        
    return d



def calc_angles_tidal(d):
    # get gas particle velocity
    v = np.array([d.vx,d.vy,d.vz])

    # instead of CGM velocity, get vector pointing from satellite to host (i.e. host position in the satellite rest frame) 
    r_sat =np.array([d.sat_Xc,d.sat_Yc,d.sat_Zc])
    r_host = np.array([d.host_Xc,d.host_Yc,d.host_Zc])
    r_rel = r_host - r_sat

    # take the dot product and get the angle, in degrees
    v_hat = v / np.linalg.norm(v)
    r_rel_hat = r_rel / np.linalg.norm(r_rel)
    angle = np.arccos(np.dot(v_hat,r_rel_hat)) * 180/np.pi

    d['angle_tidal'] = angle
        
    return d


def calc_ejected_expelled(sim, haloid, save=True, verbose=True):
    import tqdm
    data = read_tracked_particles(sim, haloid, verbose=verbose)

    if verbose: print(f'Now computing ejected/expelled particles for {sim}-{haloid}...')
    ejected = pd.DataFrame()
    cooled = pd.DataFrame()
    expelled = pd.DataFrame()
    accreted = pd.DataFrame()
    
    pids = np.unique(data.pid)
    for pid in tqdm.tqdm(pids):
        dat = data[data.pid==pid]

        sat_disk = np.array(dat.sat_disk, dtype=bool)
        sat_halo = np.array(dat.sat_halo, dtype=bool)
        in_sat = np.array(dat.in_sat, dtype=bool)
        outside_sat = ~in_sat

        host_halo = np.array(dat.host_halo, dtype=bool)
        host_disk = np.array(dat.host_disk, dtype=bool)
        IGM = np.array(dat.IGM, dtype=bool)
        other_sat = np.array(dat.other_sat, dtype=bool)
        
        time = np.array(dat.time,dtype=float)



        for i,t2 in enumerate(time[1:]):
                i += 1
                if sat_disk[i-1] and sat_halo[i]:
                    out = dat[time==t2].copy()
                    ejected = pd.concat([ejected, out])
                    
                if sat_halo[i-1] and sat_disk[i]:
                    out = dat[time==t2].copy()
                    cooled = pd.concat([cooled, out])
                    
                if in_sat[i-1] and outside_sat[i]:
                    out = dat[time==t2].copy()
                    if sat_halo[i-1]:
                        out['state1'] = 'sat_halo'
                    elif sat_disk[i-1]:
                        out['state1'] = 'sat_disk'
                        
                    expelled = pd.concat([expelled, out])
                    
                if outside_sat[i-1] and in_sat[i]:
                    out = dat[time==t2].copy()
                    if sat_halo[i]:
                        out['state2'] = 'sat_halo'
                    elif sat_disk[i]:
                        out['state2'] = 'sat_disk'
                        
                    accreted = pd.concat([accreted, out])

    # apply the calc_angles function along the rows of ejected and expelled
    print('Calculating ejection angles')
    ejected = ejected.apply(calc_angles, axis=1)
    print('Calculating expulsion angles')
    expelled = expelled.apply(calc_angles, axis=1)
    
#     # apply the calc_angles function along the rows of ejected and expelled
#     print('Calculating ejection angles (for tidal force)')
#     ejected = ejected.apply(calc_angles_tidal, axis=1)
#     print('Calculating expulsion angles (for tidal force)')
#     expelled = expelled.apply(calc_angles_tidal, axis=1)
    
    if save:
        key = f'{sim}_{str(int(haloid))}'
        filepath = '../../Data/ejected_particles.hdf5'
        print(f'Saving {key} ejected particle dataset to {filepath}')
        ejected.to_hdf(filepath, key=key)
        
        filepath = '../../Data/cooled_particles.hdf5'
        print(f'Saving {key} cooled particle dataset to {filepath}')
        cooled.to_hdf(filepath, key=key)

        filepath = '../../Data/expelled_particles.hdf5'
        print(f'Saving {key} expelled particle dataset to {filepath}')
        expelled.to_hdf(filepath, key=key)
                
        filepath = '../../Data/accreted_particles.hdf5'
        print(f'Saving {key} accreted particle dataset to {filepath}')
        accreted.to_hdf(filepath, key=key)
        
        
    print(f'Returning (ejected, cooled, expelled, accreted) datasets...')

    return ejected, cooled, expelled, accreted
        

def read_ejected_expelled(sim, haloid):
    key = f'{sim}_{str(int(haloid))}'
    ejected = pd.read_hdf('../../Data/ejected_particles.hdf5', key=key)
    cooled = pd.read_hdf('../../Data/cooled_particles.hdf5', key=key)
    expelled = pd.read_hdf('../../Data/expelled_particles.hdf5', key=key)
    accreted = pd.read_hdf('../../Data/accreted_particles.hdf5', key=key)
    print(f'Returning (ejected, cooled, expelled, accreted) for {sim}-{haloid}...')
    return ejected, cooled, expelled, accreted
        
    
def read_all_ejected_expelled():
    ejected = pd.DataFrame()
    cooled = pd.DataFrame()
    expelled = pd.DataFrame()
    accreted = pd.DataFrame()
    keys = get_keys()
    for key in keys:
        if key in ['h148_3','h148_28','h242_12']: continue;
            
        ejected1 = pd.read_hdf('../../Data/ejected_particles.hdf5', key=key)
        ejected1['key'] = key
        ejected = pd.concat([ejected, ejected1])
        cooled1 = pd.read_hdf('../../Data/cooled_particles.hdf5', key=key)
        cooled1['key'] = key
        cooled = pd.concat([cooled, cooled1])
        expelled1 = pd.read_hdf('../../Data/expelled_particles.hdf5', key=key)
        expelled1['key'] = key
        expelled = pd.concat([expelled, expelled1])
        accreted1 = pd.read_hdf('../../Data/accreted_particles.hdf5', key=key)
        accreted1['key'] = key
        accreted = pd.concat([accreted, accreted1])

    print(f'Returning (ejected, cooled, expelled, accreted) for all available satellites...')
    return ejected, cooled, expelled, accreted

