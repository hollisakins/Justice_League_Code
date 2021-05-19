# This script is intended to calculate satellite SFHs using the following method:
# 1. get an array for formation times for stars in the satellite at z=0
# 2. get an array of z=0 masses for stars in the satellite at z=0
# 3. use fsps to convert those z=0 masses to formation masses 
# 4. use that information to get SFR as a function of time (histogram of tform weighted by massform)
# 5. fit a spline curve to the stellar mass of the satellite as a function of time
# 6. divide the SFR by the interpolated stellar mass to get a specific star formation history


# is this better than just using sSFR vs. time like I did in paper 1? higher time resolution... but might be missing some things?

import pynbody
import numpy as np
import sys
import pickle
import pandas as pd
from scipy.interpolate import UnivariateSpline
age = 13.800797497330507

def read_timesteps(simname):
    '''Function to read in the timestep bulk-processing datafile (from /home/akinhol/Data/Timescales/DataFiles/{name}.data)'''
    data = []
    with open(f'../../Data/timesteps_data/{simname}.data','rb') as f:
        while True:
            try: 
                data.append(pickle.load(f))
            except EOFError:
                break
    
    data = pd.DataFrame(data)
    return data

sim = str(sys.argv[1])
haloid = int(sys.argv[2])

with open('../../Data/filepaths_haloids.pickle','rb') as f:
    d = pickle.load(f)
    filepaths = d['filepaths'][sim]
    filepath = filepaths[0]

print('Loading sim...')
s = pynbody.load(filepath)
s.physical_units()
h = s.halos()
halo = h[haloid]

stars = halo.stars

tform = np.array(stars['tform'])
### THIS IS WHERE YOU WOULD NEED TO PUT FSPS CODE TO CONVERT MASS INTO MASSFORM
massform = np.array(stars['mass'])
# bins = np.linspace(0, age, 100)
# bincenters = 0.5*(bins[1:]+bins[:-1])

# sfr, bins = np.histogram(tform, weights=massform, bins=bins)


# get timestep data to spline fit mstar
# timesteps = read_timesteps(sim)
# timesteps = timesteps[timesteps.z0haloid==haloid]
# time = np.array(timesteps.time,dtype=float)
# mstar = np.log10(np.array(timesteps.mstar,dtype=float))

# spl = UnivariateSpline(time,mstar)
# mstar_int = spl(bincenters)

# get sSFR
# sSFR = sfr/np.power(10,mstar_int)


outdict = {'tform':tform, 'massform':massform}
out = pd.DataFrame(outdict)
out.to_hdf('../../Data/sfhs.hdf5', key=str(sim)+'_'+str(haloid))









