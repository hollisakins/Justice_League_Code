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

def read_ram_pressure(sim, haloid):
    timescales = read_timescales()
    path = '../../Data/ram_pressure.hdf5'
    key = f'{sim}_{haloid}'

    data = pd.read_hdf(path, key=key)
    data['Pram_adv'] = np.array(data.Pram_adv,dtype=float)
    data['Pram'] = np.array(data.Pram,dtype=float)
    data['Prest'] = np.array(data.Prest,dtype=float)
    data['ratio'] = data.Pram_adv / data.Prest
    dt = np.array(data.t)[1:] - np.array(data.t)[:-1]
    dt = np.append(dt[0],dt)
    data['dt'] = dt
    
    timescales = read_timescales()
    ts = timescales[(timescales.sim==sim)&(timescales.haloid==haloid)]
    data['tau'] = ts.tinfall.iloc[0] - ts.tquench.iloc[0]    
    data['tquench'] = age - ts.tquench.iloc[0]   

    # load ejected/expelled data
    ejected,cooled,expelled,accreted = read_ejected_expelled(sim, haloid)

    Mgas_div = np.array(data.M_gas,dtype=float)
    Mgas_div = np.append(Mgas_div[0], Mgas_div[:-1])
    data['Mgas_div'] = Mgas_div
    
    particles = read_tracked_particles(sim,haloid)
    particles['m_disk'] = np.array(particles.mass,dtype=float)*np.array(particles.sat_disk,dtype=int)
    
    data = pd.merge_asof(data, particles.groupby(['time']).m_disk.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'m_disk':'M_disk'})
    
    Mdisk_div = np.array(data.M_disk,dtype=float)
    Mdisk_div = np.append(Mdisk_div[0], Mdisk_div[:-1])
    data['Mdisk_div'] = Mdisk_div
    
    # first, ejected gas
    data = pd.merge_asof(data, ejected.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_ejected'})
    data['Mdot_ejected'] = data.M_ejected / data.dt
    data['Mdot_ejected_by_Mgas'] = data.Mdot_ejected / Mgas_div
    data['Mdot_ejected_by_Mdisk'] = data.Mdot_ejected / Mdisk_div

    # next, cooled gas
    data = pd.merge_asof(data, cooled.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_cooled'})
    data['Mdot_cooled'] = data.M_cooled / data.dt
    data['Mdot_cooled_by_Mgas'] = data.Mdot_cooled / Mgas_div
    data['Mdot_cooled_by_Mdisk'] = data.Mdot_cooled / Mdisk_div

    # next, expelled gas
    expelled_disk = expelled[expelled.state1 == 'sat_disk']
    expelled_th30 = expelled[expelled.angle <= 30]
    
    data = pd.merge_asof(data, expelled.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_expelled'})
    data['Mdot_expelled'] = data.M_expelled / data.dt
    data['Mdot_expelled_by_Mgas'] = data.Mdot_expelled / Mgas_div

    data = pd.merge_asof(data, expelled_disk.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_expelled_disk'})
    data['Mdot_expelled_disk'] = data.M_expelled_disk / data.dt
    data['Mdot_expelled_disk_by_Mgas'] = data.Mdot_expelled_disk / Mgas_div
    data['Mdot_expelled_disk_by_Mdisk'] = data.Mdot_expelled_disk / Mdisk_div

    data = pd.merge_asof(data, expelled_th30.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_expelled_th30'})
    data['Mdot_expelled_th30'] = data.M_expelled_th30 / data.dt
    data['Mdot_expelled_th30_by_Mgas'] = data.Mdot_expelled_th30 / Mgas_div
    
    # finally, accreted gas
    accreted_disk = accreted[accreted.state2 == 'sat_disk']
    
    data = pd.merge_asof(data, accreted.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_accreted'})
    data['Mdot_accreted'] = data.M_accreted / data.dt
    data['Mdot_accreted_by_Mgas'] = data.Mdot_accreted / Mgas_div

    data = pd.merge_asof(data, accreted_disk.groupby(['time']).mass.sum().reset_index(), left_on='t', right_on='time')
    data = data.rename(columns={'mass':'M_accreted_disk'})
    data['Mdot_accreted_disk'] = data.M_accreted_disk / data.dt
    data['Mdot_accreted_disk_by_Mgas'] = data.Mdot_accreted_disk / Mgas_div
    data['Mdot_accreted_disk_by_Mdisk'] = data.Mdot_accreted_disk / Mdisk_div

    
    dM_gas = np.array(data.M_gas,dtype=float)[1:] - np.array(data.M_gas,dtype=float)[:-1]
    dM_gas = np.append([np.nan],dM_gas)
    data['Mdot_gas'] = dM_gas / np.array(data.dt)
    
    
    dM_disk = np.array(data.M_disk,dtype=float)[1:] - np.array(data.M_disk,dtype=float)[:-1]
    dM_disk = np.append([np.nan],dM_disk)
    data['Mdot_disk'] = dM_disk / np.array(data.dt)
    
    data['key'] = key
    
    M_gas_init = np.array(data.M_gas)[np.argmin(data.t)]
    data['f_gas'] = np.array(data.M_gas)/M_gas_init
    
    return data
 
    
def read_all_ram_pressure():
    data_all = pd.DataFrame()
    
    keys = ['h148_13','h148_28','h148_37','h148_45','h148_68','h148_80','h148_283',
            'h148_278','h148_329','h229_20','h229_22','h229_23','h229_27','h229_55',
            'h242_24','h242_41','h242_80','h329_33','h329_137']
    
    i = 1
    for key in keys:
        print(i, end=' ')
        i += 1
        sim = key[:4]
        haloid = int(key[5:])
        data = read_ram_pressure(sim, haloid)
        data_all = pd.concat([data_all,data])  
    
    return data_all