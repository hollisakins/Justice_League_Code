# The purpose of this file is to perform a series of data manipuation and processing commands to particle tracking data in bulk. 
# In particular, functions in this file import particle tracking and ram pressure data, join them as necessary, calculate kinetic 
# and potential energies of particles, classify particles as disk vs. halo, identify ejected or expulsed particles, and more. 
# The reason these functions are written here is so that we can ensure that we are using the same data processing procedures throughout 
# the analysis and not have to repeat this code for each analysis component. 
import numpy as np
import pandas as pd


def read_tracked_particles(sim, haloid):
    
    key = f'{sim}_{str(int(haloid))}'

    # import the tracked particles dataset
    path = '../../Data/tracked_particles.hdf5'
    particles = pd.read_hdf(path, key=key)

    # import the ram pressure dataset 
    path = '../../Data/ram_pressure.hdf5'
    properties = pd.read_hdf(path, key=key)

    # from the RP dataset, isolate the baryonic disk mass as a func. of time
    M = np.array(properties.M_star,dtype=float) + np.array(properties.M_gas,dtype=float)
    t = np.array(properties.t, dtype=float)
    masses = pd.DataFrame({'time':t, 'M':M})

    # merge the two together on a left join so that we get the masses with the particles
    data = pd.merge(particles, masses, on='time', how='left')

    # calculate kinetic energy of each particle
    m = np.array(data.mass) * 1.989e30
    v = np.array(data.v) * 1000
    K = 0.5 * m * v * v

    # and potential energy
    M = np.array(data.M) * 1.989e30
    r = np.array(data.r) * 3.086e19
    U = 6.6743e-11 * M * m / r

    ratio = K/U

    data['K'] = K
    data['U'] = U
    data['ratio'] = ratio

    # remove the simple classifications from the dataset, to replace with better ones
    data = data.drop(columns=['sat_disk', 'sat_halo', 'IGM', 'host_halo', 'host_disk', 'classification',])
    
    thermo_disk = (np.array(data.temp) < 1.2e4) & (np.array(data.rho) > 0.1) # thermodynamic disk
    grav_disk = np.array(data.ratio) < 1 # graviationally bound disk 
    in_satellite = np.array(data.r_per_Rvir) < 1 

    cool_disk = thermo_disk & grav_disk & in_satellite
    hot_disk = ~thermo_disk & grav_disk & in_satellite
    cool_halo = thermo_disk & ~grav_disk & in_satellite
    hot_halo = ~thermo_disk & ~grav_disk & in_satellite
    CGM = ~in_satellite
    c = cool_disk + 2*hot_disk + 3*cool_halo + 4*hot_halo + 5*CGM

    data['cool_disk'] = cool_disk
    data['hot_disk'] = hot_disk
    data['cool_halo'] = cool_halo
    data['hot_halo'] = hot_halo
    data['CGM'] = CGM
    data['c'] = c



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

def calc_ejected_expelled(sim, haloid):
    import tqdm
    data = read_tracked_particles(sim, haloid)

    ejected = pd.DataFrame()
    expelled = pd.DataFrame()

    pids = np.unique(data.pid)
    for pid in tqdm.tqdm(pids):
        dat = data[data.pid==pid]

        cool_disk = np.array(dat.cool_disk, dtype=bool)
        hot_disk = np.array(dat.hot_disk, dtype=bool)
        cool_halo = np.array(dat.cool_halo, dtype=bool)
        hot_halo = np.array(dat.hot_halo, dtype=bool)
        CGM = np.array(dat.CGM, dtype=bool)
        time = np.array(dat.time,dtype=float)
        
        can_be_expelled = False

        for i,t2 in enumerate(time[1:]):
                i += 1
                # t1 = time[i-1]
                
                if cool_disk[i-1] and (cool_halo[i] or hot_halo[i]):
                    state1 = 'cool disk'
                    out = dat[time==t2].copy()
                    out['state1'] = state1
                    ejected = pd.concat([ejected, out])
                    can_be_expelled = True
                elif hot_disk[i-1] and (cool_halo[i] or hot_halo[i]):
                    state1 = 'hot disk'
                    out = dat[time==t2].copy()
                    out['state1'] = state1
                    ejected = pd.concat([ejected, out])
                    can_be_expelled = True
                
                if can_be_expelled:
                    if (cool_halo[i-1] or hot_halo[i-1]) and CGM[i]:    
                        expelled = pd.concat([expelled, dat[time==t2]])


    # apply the calc_angles function along the rows of ejected and expelled
    print('Calculating ejection angles')
    ejected = ejected.apply(calc_angles, axis=1)
    print('Calculating expulsion angles')
    expelled = expelled.apply(calc_angles, axis=1)

    return ejected, expelled
        
                
