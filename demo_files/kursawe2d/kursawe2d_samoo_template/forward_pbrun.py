import sys
import math
import os
import pandas as pd
import numpy as np

def kursawe_2d(x):
    x = np.array(x)
    obj1 = -10 * np.exp(-0.2 * np.sqrt(x[0]**2 + x[1]**2))
    obj2 = np.abs(x[0])**0.8 + 5 * np.sin(x[0]**3) + np.abs(x[1])**0.8 + 5 * np.sin(x[1]**3)
    return obj1, obj2

def helper(pvals=None):
    if pvals is None:
        x = pd.read_csv("dv.dat").values.reshape(-1).tolist()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        x = np.array(list(pvals_ordered.values()))
    
    obj1, obj2 = kursawe_2d(x)
    sim = {"obj1": obj1, "obj2": obj2,}
    
    with open('output.dat','w') as f:
        f.write('obsnme,obsval\n')
        f.write('obj1,'+str(sim["obj1"])+'\n')
        f.write('obj2,'+str(sim["obj2"])+'\n')
    return sim


def ppw_worker(pst_name,host,port):
    import pyemu
    ppw = pyemu.os_utils.PyPestWorker(pst_name,host,port,verbose=False)
    pvals = ppw.get_parameters()
    if pvals is None:
        return

    obs = ppw._pst.observation_data.copy()
    obs = obs.loc[ppw.obs_names,"obsval"]

    while True:

        sim = helper(pvals=pvals)

        obs.update(sim)
        
        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    helper()
