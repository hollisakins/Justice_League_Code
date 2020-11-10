# Ram Pressure Stripping

Here is where I will put code to analyze ram pressure stripping and quenching of dwarf satellites. 

This goal of this code is to efficiently and systematically analyze the flow of gas through dwarf satellites as they quench. 


## Summary of code so far

> particletracking.py 

This script runs particle tracking on a specified satellite in a specified simulation, e.g. `python particletracking.py h329 11`.

The code tracks gas starting at the snapshot where the satellite first crosses $2~R_{\rm vir}$ and ending where at $z=0$. 
Gas particles that are tracked are *those that are in the satellite for at least one snapshot in this range*. 


Data is stored at `/Data/{sim}_{z0haloid}_particles.hdf5`. The HDF5 file type can be read into python easily as a `pandas` DataFrame using `pandas.read_hdf()`. 


### Option to run multiple scripts in parallel

The script `runall.sh` can be edited to run multiple particle tracking scripts in parallel, using the `&` bash syntax. I wouldn't recommend running more than 10 scripts at ocne though, as each one can use 5% of the memory on quirm. 

