import pynbody
import pandas as pd
import numpy as np
import pickle
import sys
import tqdm
import os

hubble =  0.6776942783267969

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

    dist = np.array(ts.h1dist, dtype=float) # doesn't need scale factor correction since its r/Rvir
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

    output = pd.DataFrame()
    print('Starting tracking/analysis...')
    scrap = True
    for f,haloid,h1id in tqdm.tqdm(zip(filepaths,haloids,h1ids),total=len(filepaths)):
        s = pynbody.load(f)
        s.physical_units()
        
        igasords = np.array(s.s['igasorder'],dtype=int)
        iordsStars = np.array(s.s['iord'],dtype=int)
        
        formedBool = np.isin(igasords,iords) # boolean array describing whether that star particle in the sim formed from one of our tracked gas particles
        alreadyTrackedBool = np.isin(iordsStars, output.pid) # boolean array describing whether we've already tracked that star particle
        
        formedStars = s.s[formedBool & ~alreadyTrackedBool] # formedStars is the star particles that formed from one of our gas particles and that we haven't already tracked
        
        # save formation times, masses, iords, and igasords of star particles that formed from gas particles we're tracking
        output = pd.concat([output, analysis(formedStars, scrap)])
        
        scrap = False
    
    return output


def analysis(formedStars, scrap):
    output = pd.DataFrame()
    a = float(s.properties['a'])

    # calculate properties that are invariant to centering
    output['tform'] = np.array(formedStars.s['tform'].in_units('Gyr'), dtype=float)
    output['pid'] = np.array(formedStars.s['iord'],dtype=int)
    output['igasorder'] = np.array(formedStars.s['igasorder'],dtype=int)
    
    star_masses = np.array(formedStars.s['mass'].in_units('Msol'),dtype=float)
    star_metals = np.array(formedStars.s['metals'],dtype=float)
    star_ages = np.array(formedStars.s['age'].in_units('Myr'),dtype=float)
    size = len(star_ages)
    
    fsps_ssp = fsps.StellarPopulation(sfh=0,zcontinuous=1,imf_type=2,zred=0.,add_dust_emission=False)
    solar_Z = 0.0196
    
    print(f'\t {sim}-{z0haloid}: performing FSPS calculations on {size} star particles')
    massform = np.array([])
    for age, metallicity, mass in zip(star_ages, star_metals, star_masses):
        fsps_ssp.params['logzsol'] = np.log10(metallicity/solar_Z)
        mass_remaining = fsps_ssp.stellar_mass
        massform = np.append(massform, mass / np.interp(np.log10(age*1e9), fsps_ssp.ssp_ages, mass_remaining))
    
    output['massform'] = massform
    
    if scrap:
        output['formed_during_tracking_period'] = np.array([False]*len(formedStars))
    else:
        output['formed_during_tracking_period'] = np.array([True]*len(formedStars))

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
        h1ids = np.flip(h1ids[:snap_start+1])
        
    # filepaths and haloids now go the "right" way, i.e. starts from start_snap and goes until z=0
    assert len(filepaths) == len(haloids)
    assert len(haloids) == len(h1ids)

    # we save the data as an .hdf5 file since this is meant for large datasets, so that should work pretty good
    output = run_tracking(sim, z0haloid, filepaths, haloids, h1ids)
    output.to_hdf('../../Data/tracked_stars.hdf5',key=f'{sim}_{z0haloid}')







    





                
