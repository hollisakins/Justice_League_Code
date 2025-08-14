#!/bin/bash

# cd ~/Research/Justice_League_Code/Analysis/RamPressure/
pwd
date

python rampressure.py h242 38 & # 1
python rampressure.py h329 29 & # 2
python rampressure.py h229 22 & # 3
python rampressure.py h148 12 & # 4
wait
python rampressure.py h229 14 & # 5
python rampressure.py h242 21 & # 6
python rampressure.py h229 18 & # 7
python rampressure.py h242 69 & # 8
wait
python rampressure.py h148 34 & # 9
python rampressure.py h148 27 & # 10
python rampressure.py h148 55 & # 11
python rampressure.py h148 251 & # 12
wait
python rampressure.py h229 20 & # 13
python rampressure.py h148 38 & # 14
python rampressure.py h148 249 & # 15
python rampressure.py h229 49 & # 16
wait
python rampressure.py h329 117 & # 17
python rampressure.py h148 65 & # 18
python rampressure.py h148 282 & # 19
wait

# include `wait` in between commands to do them in batches
# by my estimate each command uses at most 5-7% of the memory on quirm, so don't run more than 10 at once (ideally less than 10)