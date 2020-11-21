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
age = 13.800797497330507

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

hist = np.histogram(tform, bins=np.linspace(0, age, 100))
print(hist)





