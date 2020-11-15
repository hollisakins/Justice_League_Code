# Ram Pressure Stripping

In this directory lies code I'm using to analyze ram pressure stripping and quenching in dwarf galaxies. The goal of this code is to efficiently and systematically analyze the flow of gas out of (and into) dwarf satellites as they quench. 

## Code

> particletracking.py 

This script runs particle tracking on a particular satellite in a particular simulation, which are specified at the command line (e.g. `python particletracking.py h329 11`). The code tracks gas particles, starting at the snapshot where the satellite first crosses 2 Rvir from its host and ending at redshift 0. Gas particles that are tracked are *those that are in the satellite for at least one snapshot in this range*. 

> plot_gen.py

This script generates (and saves) plots of tracked gas particles and satellite orbits for a simulation/satellite specified as a bash argument. 

> runall.sh

This script is used to run multiple particle tracking scripts in parallel, using the `&` bash syntax. This can greatly speed up the process of running particle tracking on multiple galaxies, but also uses a lot more computational resources. I wouldn't recommend running more than 5 at a time on `quirm`. 

## Data

The `particletracking.py` script draws data directly from the simulation snapshots. To speed up the process of analyzing these satellites over time, I have stored the simulation snapshot filepaths and main progenitor haloids for each redshift 0 satellite at `/Data/filepaths_haloids.pickle`. The scripts in this directory utilize this pickle file to get satellite haloid information. 

Data output from `particletracking.py` is stored at `/Data/tracked_particles.hdf5`. The HDF5 file format stores the output data efficiently and allows for data from all the galaixes to be stored in one file. The HDF5 file type can be read into python easily as a `pandas` DataFrame using `pandas.read_hdf(filepath, key=key)`, where the `key` option specifies the table you want. Each galaxy is saved with a key identifying its host and redshift 0 haloid, i.e. `key='h148_13'`. 

The `tracked_particles.hdf5` file is not stored on Github (as it is too large) but can be found on `quirm` at 

> /home/akinshol/Research/Justice_League_Code/Data/tracked_particles.hdf5


