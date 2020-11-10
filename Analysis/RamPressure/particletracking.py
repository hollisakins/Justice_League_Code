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


# loop through all the snapshots we're interested in first and get the iords of all gas particles in the halo at each snapshot
# then go back and loop again, this time tracking all gas particles in our iord list i.e.
# gas_particles = halo.gas[halo.gas['iord']==iord_stored]
# then you can just track those from the beginning and not have to worry about adding new particles to the tracking as you go


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


def get_iords(sim, z0haloid, filepaths, haloids):
    # '''Get the particle indices (iords) for all gas particles that have been in the halo since snap_start.''''
    path = f'../../Data/iords/{sim}_{z0haloid}.pickle'
    if os.path.exists(path):
        print(f'Found iords file at {path}, loading these...')
        with open(path,'rb') as infile:
            iords = pickle.load(infile)
    
    else:
        print(f'No iords file, computing iords to track (saving to {path})...')
        iords = np.array([])
        for f,haloid in tqdm.tqdm(zip(filepaths,haloids),total=len(filepaths)):
            s = pynbody.load(f)
            s.physical_units()
            h = s.halos()
            halo = h[haloid]
            iord = np.array(halo.gas['iord'], dtype=int)
            iords = np.union1d(iords, iord)
        
        with open(path,'wb') as outfile:
            pickle.dump(iords,outfile)

    return iords



def run_tracking(sim, z0haloid, filepaths,haloids,h1ids):
    # now we need to start tracking, so we need to get the iords
    iords = get_iords(sim, z0haloid, filepaths, haloids)
    
    use_iords = True
    verbose = False
    output = pd.DataFrame()
    print('Starting tracking/analysis...')
    for f,haloid,h1id in tqdm.tqdm(zip(filepaths,haloids,h1ids),total=len(filepaths)):
        s = pynbody.load(f)
        s.physical_units()
        h = s.halos()
        halo = h[haloid]
        h1 = h[h1id]
        snapnum = f[-4:]

        if use_iords:
            if verbose: print(f'First snapshot ({snapnum}), getting gas particles from iords')
            iord = np.array(s.gas['iord'],dtype=float)
            gas_particles = s.gas[np.isin(iord,iords)]
            use_iords = False
        else:
            if verbose: print(f'Linking snapshots {snapnum_prev} and {snapnum} with bridge object...')
            b = pynbody.bridge.OrderBridge(s_prev,s,allow_family_change=True)
            gas_particles = b(gas_particles_prev)

        # run analysis on gas particles!
        # this calls the analysis function i've defined, and concatenates the output from this snapshot to the output overall
        output = pd.concat([output, analysis(s,halo,h1,gas_particles)])

        gas_particles_prev = gas_particles
        snapnum_prev = snapnum
        s_prev = s
    
    return output


def analysis(s,halo,h1,gas_particles):
    output = pd.DataFrame()

    if len(gas_particles) != len(gas_particles.g):
        raise Exception('Some particles are no longer gas particles...')


    output['time'] = np.array([float(s.properties['time'].in_units('Gyr'))]*len(gas_particles))
    output['pid'] = np.array(gas_particles['iord'],dtype=int)
    output['rho'] = np.array(gas_particles.g['rho'].in_units('Msol kpc**-3'), dtype=float) * 4.077603812e-8 # multiply to convert to amu/cm^3
    output['temp'] = np.array(gas_particles.g['temp'].in_units('K'), dtype=float)
    output['mass'] = np.array(gas_particles.g['mass'].in_units('Msol'), dtype=float)
    output['coolontime'] = np.array(gas_particles.g['coolontime'].in_units('Gyr'),dtype=float)
    
    pynbody.analysis.halo.center(halo)
    x,y,z = gas_particles['x'],gas_particles['y'],gas_particles['z']
    Rvir = halo.properties['Rvir']/hubble
    output['r'] = np.array(np.sqrt(x**2 + y**2 + z**2), dtype=float)
    output['r_per_Rvir'] = output.r / Rvir
    output['x'] = x
    output['y'] = y
    output['z'] = z

    output['vx'] = np.array(gas_particles['vx'].in_units('km s**-1'),dtype=float)
    output['vy'] = np.array(gas_particles['vy'].in_units('km s**-1'),dtype=float)
    output['vz'] = np.array(gas_particles['vz'].in_units('km s**-1'),dtype=float)
    output['v'] = np.array(np.sqrt(output.vx**2 + output.vy**2 + output.vz**2))

    pynbody.analysis.halo.center(h1)
    x,y,z = gas_particles['x'],gas_particles['y'],gas_particles['z']
    Rvir = h1.properties['Rvir']/hubble
    output['h1dist'] = np.array(np.sqrt(x**2 + y**2 + z**2), dtype=float) / Rvir


    # classifications: sat disk, sat halo, host halo, other satellite, IGM
    sat_disk = (output.rho >= 0.1) & (output.temp <= 1.2e4) & (output.r < 3)
    sat_halo = output.r_per_Rvir < 1
    IGM = output.h1dist > 1
    host_halo = (output.r_per_Rvir > 1) & (output.h1dist < 1)
    host_disk = (output.rho >= 0.1) & (output.temp <= 1.2e4) & (output.r_per_Rvir > 1) & (output.h1dist < 0.1)
    # other satellite: how to connect AHF membership to particles?

    output['sat_disk'] = np.array(sat_disk,dtype=bool)
    output['sat_halo'] = np.array(sat_halo,dtype=bool)
    output['IGM'] = np.array(IGM,dtype=bool)
    output['host_halo'] = np.array(host_halo,dtype=bool)
    output['host_disk'] = np.array(host_disk,dtype=bool)
    classification = np.zeros(shape=sat_disk.shape)
    classification[sat_disk] = 1
    classification[sat_halo] = 2
    classification[host_disk] = 3
    classification[host_halo] = 4
    classification[IGM] = 5
    output['classification'] = classification

    return output


if __name__ == '__main__':
    sim = str(sys.argv[1])
    z0haloid = int(sys.argv[2])
    
    snap_start = get_snap_start(sim,z0haloid)
    filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,z0haloid)
    # filepaths starts with z=0 and goes to z=15 or so

    # fix the case where the satellite doesn't have merger info prior to 
    if len(haloids) < snap_start:
        snap_start = len(haloids)
        raise Exception('Careful! You may have an error since the satellite doesnt have mergertree info out to the time where you want to start. This case is untested')
    
    if len(haloids) >= snap_start:
        filepaths = np.flip(filepaths[:snap_start+1])
        haloids = np.flip(haloids[:snap_start+1])

    # filepaths and haloids now go the "right" way, i.e. starts from start_snap and goes until z=0

    # we save the data as an .hdf5 file since this is meant for large datasets, so that should work pretty good
    output = run_tracking(sim, z0haloid, filepaths, haloids, h1ids)
    output.to_hdf('../../Data/tracked_particles.hdf5',key=f'{sim}_{z0haloid}')







    





                
