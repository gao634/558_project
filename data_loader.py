import os
import numpy as np

def load_env(folder='.\env', env_id=0):
    obs=np.fromfile(folder+'env_'+env_id,np.int32)
    return obs