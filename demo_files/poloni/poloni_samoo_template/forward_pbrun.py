import sys
import math
import os
import pandas as pd
import numpy as np

def poloni(x):
    x = np.array(x)
    # Poloni's two objective function
    A1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
    A2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)
    
    B1 = 0.5 * np.sin(x[0]) - 2 * np.cos(x[0]) + np.sin(x[1]) - 1.5 * np.cos(x[1])
    B2 = 1.5 * np.sin(x[0]) - np.cos(x[0]) + 2 * np.sin(x[1]) - 0.5 * np.cos(x[1])
    
    obj1 = 1 + (A1 - B1)**2 + (A2 - B2)**2
    obj2 = (x[0] + 3)**2 + (x[1] + 1)**2
    
    return obj1, obj2

def helper(pvals=None):
    if pvals is None:
        x = pd.read_csv("dv.dat").values.reshape(-1).tolist()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        x = np.array(list(pvals_ordered.values()))
    
    obj1, obj2 = poloni(x)
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
