# Bulk Processing

When analyzing simulations, one rapidly encounters a fundamental problem: the analysis techniques that are useful can take a lot of to process on such large simulations. For example, to calculate the star-formation histories of every satellite in the Justice League simulations is not a quick task, even for a powerful computer. 

So, to workaround this, we write code in `.py` scripts that reads in the *raw* simulation files and returns a compiled dataset of relevant satellite properties. That is, we calculate these properties *in bulk*. These scripts can take as much as 12-18 hours to run, depending on what they are doing (and how efficiently the code is written). 

These bulk processing scripts produce the `.data` files that much of the analysis code uses. The `.data` extension is a convention used whenever we "pickle" data---that is, whenever we use the Python `pickle` module to store data. The `pickle` module is able to save Python objects to a file and reopen them without losing the same data format---you don't have to worry about whether the obeject was a list, array, SimArray, etc. 

The bulk processing scripts in this directory generate a few different sets of data:

## Redshift Zero

The first of these scripts, `bulk_processing.py`, analyzes the redshift zero snapshots of the eight simulations (4 Justice League, 4 Marvel). 



## Timestep Bulk Processing

The primary script here is ``timescales_bulk.py`. 





