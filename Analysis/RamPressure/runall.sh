#!/bin/bash

cd ~/Research/Justice_League_Code/Analysis/RamPressure/
pwd
date

python particletracking.py h329 11 & 
python particletracking.py h329 31 & 
...
# you get the picture

# include `wait` in between commands to do them in batches
# by my estimate each command uses at most 5-7% of the memory on quirm, so don't run more than 10 at once (ideally less than 10)
