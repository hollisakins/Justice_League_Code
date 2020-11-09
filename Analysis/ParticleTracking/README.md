# Particle Tracking

> Important Note: As of 11/8/2020 this code will not be updated. Further analysis using particle tracking will be done in `Analysis/RamPressure`

*Notebook: `ParticleTracking.ipynb`.*

The code in this notebook is really just a proof of concept for the particle tracking code that ended up in `.py` scripts. 
This code is not completely operational (I haven't been able to get particle tracking to follow gas particles that turn into stars).

*Notebook: `Stripping.ipynb`.*

The code in this notebooks takes in the data stored in `/Data/stripping_data/` and makes some basic plots in order to determine what happens to the hot/cold gas particles after infall. 

*Python scripts: `hot_gas_particle_tracking.py` and `cold_gas_particle_tracking.py`*

The code in these scripts performs particle tracking on a selection of satellites from infall until $z=0$. The resulting data is stored in `/Data/stripping_data/`.

## Data used

These notebooks use the data in `/Data/stripping_data/`, which includes information about the hot and cold gas that exists in the satellite at infall---where does it go after infall (the halo, the disk, or stripped). 
The file `/Data/stripping_data/HotGasTracking.data` also includes several properties of the satellites over time and at infall, e.g. the relative velocity, angle of infall, gas mass, halo mass, etc. 





