import numpy as np
import pandas as pd
from bulk import *

keys = get_keys()
for key in keys:
    if key=='h148_28' or key=='h148_3' or key=='h242_12': continue;
    sim = str(key[:4])
    haloid = int(key[5:])
    ejected, cooled, expelled, accreted = calc_ejected_expelled(sim,haloid, verbose=True, save=True)