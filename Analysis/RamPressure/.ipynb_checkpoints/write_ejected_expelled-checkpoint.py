import numpy as np
import pandas as pd
from bulk import *

keys = get_keys()
for key in keys:
    sim = str(key[:4])
    haloid = int(key[5:])
    ejected, expelled = calc_ejected_expelled(sim,haloid, verbose=True, save=True)