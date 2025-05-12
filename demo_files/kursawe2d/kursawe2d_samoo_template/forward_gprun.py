import pandas as pd
import numpy as np
import glob
import os
import laGPy as gpr
from scipy.stats import norm

temp_dir = "template_inner"
outer_dirs = sorted(glob.glob("outer_*"), key=lambda x: int(os.path.basename(x).split("_")[1]))

def emulate(pvals = None):
    if pvals is None:
        decvar = pd.read_csv(os.path.join(temp_dir, "dv.dat")).values.transpose()
    else:
        pvals_ordered = {pval: pvals[pval] for pval in sorted(pvals.index, key=lambda x: int(x[1:]))}
        decvar = np.array(list(pvals_ordered.values())).transpose()
    
    X = pd.read_csv(os.path.join(temp_dir, "gp_0.dv_training.csv")).drop(columns=['real_name']).values
    obj1 = pd.read_csv(os.path.join(temp_dir, "gp_0.obs_training.csv")).drop(columns=['real_name'])['obj1'].values
    obj2 = pd.read_csv(os.path.join(temp_dir, "gp_0.obs_training.csv")).drop(columns=['real_name'])['obj2'].values
    
    pred1 = gpr.laGP(Xref=decvar, start=6, end=20, X=X, Z=obj1)
    pred2 = gpr.laGP(Xref=decvar, start=6, end=20, X=X, Z=obj2)
    sim = {
        'obj1': pred1["mean"].item(),
        'obj2': pred2["mean"].item(),
        'obj1_sd': np.sqrt(pred1["s2"].item()),
        'obj2_sd': np.sqrt(pred2["s2"].item()),
    }

    with open('output.dat','w') as f:
        f.write('obsnme,obsval\n')
        f.write('obj1,'+str(sim["obj1"])+'\n')
        f.write('obj2,'+str(sim["obj2"])+'\n')
        f.write('obj1_sd,'+str(sim["obj1_sd"])+'\n')
        f.write('obj2_sd,'+str(sim["obj2_sd"])+'\n')
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
        sim = emulate(pvals=pvals)
        obs.update(sim)
        ppw.send_observations(obs.values)
        pvals = ppw.get_parameters()
        if pvals is None:
            break

if __name__ == "__main__":
    emulate()
