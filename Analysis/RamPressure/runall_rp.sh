#!/bin/bash

cd ~/Research/Justice_League_Code/Analysis/RamPressure/
pwd
date

python rampressure.py h148 13 & 
python rampressure.py h229 20 &
python rampressure.py h242 24 & 
python rampressure.py h148 37 &
python rampressure.py h148 28 & 
python rampressure.py h229 22 & 
python rampressure.py h242 80 &
python rampressure.py h148 68 &
wait

# include `wait` in between commands to do them in batches
# by my estimate each command uses at most 5-7% of the memory on quirm, so don't run more than 10 at once (ideally less than 10)
