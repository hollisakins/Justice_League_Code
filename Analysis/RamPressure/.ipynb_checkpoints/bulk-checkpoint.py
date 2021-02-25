# The purpose of this file is to perform a series of data manipuation and processing commands to particle tracking data in bulk. 
# In particular, functions in this file import particle tracking and ram pressure data, join them as necessary, calculate kinetic 
# and potential energies of particles, classify particles as disk vs. halo, identify ejected or expulsed particles, and more. 
# The reason these functions are written here is so that we can ensure that we are using the same data processing procedures throughout 
# the analysis and not have to repeat this code for each analysis component. 
import numpy as np
import pandas as pd
from numba import njit,jit
import tqdm
import time as time_module

def get_keys():
    path = '../../Data/tracked_particles.hdf5'
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
    sat_disk = in_sat & (np.array(data.r) <= np.array(data.r_gal))
    sat_halo = in_sat & (np.array(data.r) > np.array(data.r_gal))
    sat_cool_disk = sat_disk & thermo_disk
    sat_hot_disk = sat_disk & ~thermo_disk
    sat_cool_halo = sat_halo & thermo_disk
    sat_hot_halo = sat_halo & ~thermo_disk

    in_host = np.array(data.in_host) & ~in_sat
    host_disk = in_host & (np.array(data.r_rel_host) <= np.array(data.host_r_gal))
    host_halo = in_host & (np.array(data.r_rel_host) > np.array(data.host_r_gal))

    other_sat = np.array(data.in_other_sat)
    IGM = np.array(data.in_IGM)
    
    
    # basic classifications
    data['sat_disk'] = sat_disk
    data['sat_halo'] = sat_halo
    data['host_disk'] = host_disk
    data['host_halo'] = host_halo
    data['other_sat'] = other_sat
    data['IGM'] = IGM
    
    # more advanced classifications
    data['cool_disk'] = sat_cool_disk
    data['hot_disk'] = sat_hot_disk
    data['cool_halo'] = sat_cool_halo
    data['hot_halo'] = sat_hot_halo

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

# @njit
def run_particle_loop(pids,pids_unique,cool_disk1,hot_disk1,cool_halo1,hot_halo1,host_halo1,host_disk1,IGM1,other_sat1,time1):
    ejected_rowids = []
    ejected_state1 = []
    expelled_rowids = []
    expelled_state2 = []
    
    for pid in pids_unique:
        cool_disk = cool_disk1[pids==pid]
        hot_disk = hot_disk1[pids==pid]
        cool_halo = cool_halo1[pids==pid]
        hot_halo = hot_halo1[pids==pid]
        host_halo = host_halo1[pids==pid]
        host_disk = host_disk1[pids==pid]
        IGM = IGM1[pids==pid]
        other_sat = other_sat1[pids==pid]
        time = time1[pids==pid]
        
        i = 0
        can_be_expelled = False
        for t2 in time[1:]:
            i += 1
            
            x = np.where((pids==pid) & (time1==t2))[0][0]
            
            if cool_disk[i-1] and (cool_halo[i] or hot_halo[i]):
                state1 = 'cool disk'
                ejected_rowids.append(x)
                ejected_state1.append(state1)
                can_be_expelled = True
            elif hot_disk[i-1] and (cool_halo[i] or hot_halo[i]):
                state1 = 'hot disk'
                ejected_rowids.append(x)
                ejected_state1.append(state1)
                can_be_expelled = True
            if can_be_expelled:
                if (cool_halo[i-1] or hot_halo[i-1]) and host_halo[i]:    
                    state2 = 'host halo'
                    expelled_rowids.append(x)
                    expelled_state2.append(state2)
                    
                if (cool_halo[i-1] or hot_halo[i-1]) and host_disk[i]:    
                    state2 = 'host disk'
                    expelled_rowids.append(x)
                    expelled_state2.append(state2)
                if (cool_halo[i-1] or hot_halo[i-1]) and IGM[i]:    
                    state2 = 'IGM'
                    expelled_rowids.append(x)
                    expelled_state2.append(state2)

    return ejected_rowids, ejected_state1, expelled_rowids, expelled_state2







def calc_ejected_expelled(sim, haloid, save=True, verbose=True):
    data = read_tracked_particles(sim, haloid, verbose=verbose)

    if verbose: print(f'Now computing ejected/expelled particles for {sim}-{haloid}...')

    pids = np.array(data.pid, dtype=int)
    pids_unique = np.unique(pids)
    cool_disk = np.array(data.cool_disk, dtype=bool)
    hot_disk = np.array(data.hot_disk, dtype=bool)
    cool_halo = np.array(data.cool_halo, dtype=bool)
    hot_halo = np.array(data.hot_halo, dtype=bool)

    host_halo = np.array(data.host_halo, dtype=bool)
    host_disk = np.array(data.host_disk, dtype=bool)
    IGM = np.array(data.IGM, dtype=bool)
    other_sat = np.array(data.other_sat, dtype=bool)

    time = np.array(data.time,dtype=float)
    
    t0 = time_module.time()
    ejected_rowids, ejected_state1, expelled_rowids, expelled_state2 = run_particle_loop(pids,pids_unique,cool_disk,hot_disk,cool_halo,hot_halo,
                                                                                         host_halo,host_disk,IGM,other_sat,time)
    t1 = time_module.time()    
    total = t1-t0
    print(f'Time took: {total:.2f} sec')
    print(len(ejected_rowids))
    
    ejected = data.iloc[np.array(ejected_rowids)]
    expelled = data.iloc[np.array(expelled_rowids)]
    ejected['state1'] = np.array(ejected_state1)
    expelled['state2'] = np.array(expelled_state2)

    # apply the calc_angles function along the rows of ejected and expelled
    print('Calculating ejection angles')
    ejected = ejected.apply(calc_angles, axis=1)
    print('Calculating expulsion angles')
    expelled = expelled.apply(calc_angles, axis=1)
    
    if save:
        key = f'{sim}_{str(int(haloid))}'
        filepath = '../../Data/ejected_particles.hdf5'
        print(f'Saving {key} ejected particle dataset to {filepath}')
        ejected.to_hdf(filepath, key=key)

        filepath = '../../Data/expelled_particles.hdf5'
        print(f'Saving {key} expelled particle dataset to {filepath}')
        expelled.to_hdf(filepath, key=key)
        
        
    print(f'Returning both datasets...')

    return ejected, expelled
        

def read_ejected_expelled(sim, haloid):
    key = f'{sim}_{str(int(haloid))}'
    ejected = pd.read_hdf('../../Data/ejected_particles.hdf5', key=key)
    expelled = pd.read_hdf('../../Data/expelled_particles.hdf5', key=key)
    print(f'Returning ejected,expelled for {sim}-{haloid}...')
    return ejected, expelled
        
    
def read_all_ejected_expelled():
    ejected = pd.DataFrame()
    expelled = pd.DataFrame()
    keys = get_keys()
    for key in keys:
        ejected = pd.concat([ejected, pd.read_hdf('../../Data/ejected_particles.hdf5', key=key)])
        expelled = pd.concat([expelled, pd.read_hdf('../../Data/expelled_particles.hdf5', key=key)])
    print(f'Returning ejected,expelled for all available satellites...')
    return ejected, expelled

