import numpy as np
import pandas as pd
from bulk import calc_ejected_expelled

# keys = ['h148_13', 'h148_28', 'h148_37', 'h148_68', 'h229_20', 'h229_22', 'h242_24', 'h242_80']
sims = ['h229', 'h148','h148','h148','h229','h242','h242', 'h148']
haloids = [20, 13, 37, 68, 22, 24, 80, 28]

for sim, haloid in zip(sims,haloids):
    print(f'Running for {sim}, halo {haloid}')
    ejected, expelled = calc_ejected_expelled(sim,haloid)

    key = f'{sim}_{str(int(haloid))}'
    print('Writing to ../../Data/')
    ejected.to_hdf('../../Data/ejected_particles.hdf5',key=key)
    expelled.to_hdf('../../Data/expelled_particles.hdf5',key=key)

