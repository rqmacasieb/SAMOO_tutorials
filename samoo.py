
import pandas as pd
import os
import laGPy as gpr
import pyemu
import glob
import sys
import shutil
import datetime
from sklearn.cluster import KMeans
import numpy as np
import argparse
import re

nmax_outer = 5
nmax_inner = 20

restart = False #if continuing from last outer iter
num_workers = 8
max_infill = 50
pop_size = 50
tmpl_in = "template_inner"

# parser = argparse.ArgumentParser()
# parser.add_argument('--output-dir', type=str)
# parser.add_argument('--seed', type=int)
# #parser.add_argument('--master-host', type=str)
# args = parser.parse_args()
# port = 4100 + args.seed
# output_dir = args.output_dir
# #master_host = args.master_host

output_dir = '.'
port = 4000

def inner_opt(iitidx):
    sys.path.insert(0, tmpl_in)
    from forward_gprun import ppw_worker as ppw_function 
    pyemu.os_utils.start_workers(tmpl_in, "pestpp-mou", 
                                 "fon.pst", num_workers=num_workers, 
                                 worker_root=".", master_dir="./inner_"+str(iitidx), port=port,
                                 ppw_function=ppw_function)
    sys.path.remove(tmpl_in)

    #delete some files to save space
    file_formats_to_delete = ["*.gp", "*.trimmed.archive.summary.csv"]
    for file_format in file_formats_to_delete:
        files_to_delete = glob.glob(os.path.join(f"./inner_{iitidx}", file_format))
        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)

    return sorted([d for d in os.listdir() if d.startswith("inner_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))
    
def outer_sweep(oitidx):   
    if oitidx == 0:
        shutil.copy(os.path.join("template_outer", "gp.lhs.dv_pop.csv"), 
                    os.path.join("template_outer", "infill.dv_pop.csv"))

    sys.path.insert(0, "template_outer")
    from forward_pbrun import ppw_worker as ppw_function 
    pyemu.os_utils.start_workers("template_outer", "pestpp-mou", 
                                 "fon.pst", num_workers=num_workers, 
                                 worker_root=".", master_dir="./outer_"+str(oitidx), port=port,
                                 ppw_function=ppw_function)
    sys.path.remove("template_outer")

    return sorted([d for d in os.listdir() if d.startswith("outer_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))

def get_dirlist():
    inner_dirs = sorted([d for d in os.listdir() if d.startswith("inner_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))
    outer_dirs = sorted([d for d in os.listdir() if d.startswith("outer_") and os.path.isdir(d)], key=lambda x: int(x.split("_")[1]))

    return inner_dirs, outer_dirs

def build_training_dataset(X, Y):

    X.to_csv(os.path.join(tmpl_in, f"gp_0.dv_training.csv"), index=False)
    X = X.drop(columns=['real_name']).values

    Y.to_csv(os.path.join(tmpl_in, f"gp_0.obs_training.csv"), index=False)
    Y = Y.drop(columns=['real_name'])['func'].values

    # gpr.buildGP(X, Y, fname=os.path.join(tmpl_in, f"gp_0.gp"))
    print(f"\n{datetime.datetime.now()}: GP training dataset saved. \n")

def inner_prep(inner_dirs, outer_dirs):
    curr_dv = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/*0.dv_pop.csv", recursive=True)[0])
    curr_obs = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/*0.obs_pop.csv", recursive=True)[0])

    #copy previous outer repo update dv and obs files
    if outer_dirs[-1].endswith("_0"):      
        restart_dv = curr_dv
        restart_obs = curr_obs
    else:
        restart_dv = pd.read_csv(os.path.join(outer_dirs[-1], "outer_repo", "outer_repo.archive.dv_pop.csv"))
        restart_obs = pd.read_csv(os.path.join(outer_dirs[-1], "outer_repo", "outer_repo.archive.obs_pop.csv"))  

        dv_restart_pool = curr_dv[~curr_dv['real_name'].isin(restart_dv['real_name'].values)]
        if restart_dv.shape[0] < pop_size:
            if pop_size - restart_dv.shape[0] <= dv_restart_pool.shape[0]:
                restart_dv_subsample = dv_restart_pool.sample(n=pop_size - restart_dv.shape[0])
            else:
                restart_dv_subsample = dv_restart_pool

            restart_obs_subsample = curr_obs[curr_obs['real_name'].isin(restart_dv_subsample['real_name'])]
            restart_dv = pd.concat([restart_dv, restart_dv_subsample], ignore_index=True)
            restart_obs = pd.concat([restart_obs, restart_obs_subsample], ignore_index=True)
        else:
            restart_dv = restart_dv.sample(n=pop_size)
            restart_obs = curr_obs[curr_obs['real_name'].isin(restart_dv['real_name'])]

    restart_dv = restart_dv.loc[~restart_dv.duplicated(subset=restart_dv.columns.difference(['real_name']), keep='first')]
    restart_dv = restart_dv.sort_values('real_name')
    restart_dv.to_csv(os.path.join(tmpl_in, "initial.dv_pop.csv"), index=False)

    restart_obs = restart_obs.loc[~restart_obs.duplicated(subset=restart_obs.columns.difference(['real_name']), keep='first')]
    restart_obs = restart_obs.sort_values('real_name')
    restart_obs.to_csv(os.path.join(tmpl_in, "initial.obs_pop.csv"), index=False)

    #remove existing gp files and training data from template dir
    for file in glob.glob(os.path.join(tmpl_in, "*.gp")) + \
                glob.glob(os.path.join(tmpl_in, "*dv_training.csv")) + \
                glob.glob(os.path.join(tmpl_in, "*obs_training.csv")):
        os.remove(file)

    #update training data
    if outer_dirs[-1].endswith("_0"):
        training_dv, training_obs = curr_dv, curr_obs
    else:
        training_dv = pd.concat([pd.read_csv(glob.glob(os.path.join(inner_dirs[-1], "*0.dv_training.csv"), recursive=True)[0]), 
                                 curr_dv], ignore_index=True)
        training_obs = pd.concat([pd.read_csv(glob.glob(os.path.join(inner_dirs[-1], "*0.obs_training.csv"), recursive=True)[0]), 
                                  curr_obs], ignore_index=True)
        training_dv.columns, training_obs.columns = curr_dv.columns.values, curr_obs.columns.values

    build_training_dataset(training_dv, training_obs)

def update_outer_repo(outer_dirs):
    base_path = os.path.join(".", outer_dirs[-2])
    if len(outer_dirs) > 2:
        base_path = os.path.join(base_path, "outer_repo")
    
    prev_dv_file = glob.glob(os.path.join(base_path, "*.archive.dv_pop.csv"), recursive=True)
    prev_dv = pd.read_csv(prev_dv_file[0])
    prev_obs_file = glob.glob(os.path.join(base_path, "*.archive.obs_pop.csv"), recursive=True)
    prev_obs = pd.read_csv(prev_obs_file[0])

    curr_dv = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/*0.dv_pop.csv", recursive=True)[0])
    curr_obs = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/*0.obs_pop.csv", recursive=True)[0])

    for file_type in ["dv_pop", "obs_pop"]:
        merged_file = pd.concat([prev_dv, curr_dv] if file_type == "dv_pop" else [prev_obs, curr_obs], ignore_index=True)
        merged_file.drop_duplicates(subset='real_name', inplace=True)
        merged_file.to_csv(os.path.join(".", "template_repo_update", f"merged.{file_type}.csv"), index=False)

    #run pestpp mou in pareto sorting mode
    pyemu.os_utils.start_workers("template_repo_update", "pestpp-mou", 
                                 "outer_repo.pst", num_workers=num_workers, port=port,
                                 worker_root=".", master_dir="temp")

    #copy outer repo update files to outer dir
    subdir = os.path.join(outer_dirs[-1], "outer_repo")
    if os.path.exists(subdir):
        os.remove(subdir)
    os.mkdir(subdir)

    outer_repo_sumlist = glob.glob(os.path.join("temp", "outer_repo.pareto*"), recursive=True)
    for name in outer_repo_sumlist:
        shutil.copy(name, subdir)
        
    outer_repo_sumlist = glob.glob(os.path.join("temp", "outer_repo.archive*"), recursive=True)
    for name in outer_repo_sumlist:
        shutil.copy(name, subdir)
    
    #clean up temp directory
    shutil.rmtree("temp")

def prep_templates():
    print(f"\n{datetime.datetime.now()}: prepping templates \n")
          
    pst_files = glob.glob(os.path.join('template', '*.pst'))
    if len(pst_files) != 1:
        raise ValueError("There should be exactly one .pst file in the template directory.")
    
    #prep outer iter pst file
    print(f"\n{datetime.datetime.now()}: prepping outer template \n")
    pst = pyemu.Pst(pst_files[0])
    if os.path.exists('template_outer'):
        shutil.rmtree('template_outer')
    shutil.copytree('template', 'template_outer')

    pst.control_data.noptmax = -1
    pst.model_command = 'python forward_pbrun.py'
    pst.pestpp_options['mou_dv_population_file'] = 'infill.dv_pop.csv'
    pst.write(os.path.join('template_outer', os.path.basename(pst_files[0])))

    print(f"\n{datetime.datetime.now()}: outer template prepped \n")

    #prep outer repo update template
    print(f"\n{datetime.datetime.now()}: prepping repo update template \n")
    pst = pyemu.Pst(pst_files[0])
    if os.path.exists('template_repo_update'):
        shutil.rmtree('template_repo_update')
    shutil.copytree('template', 'template_repo_update')

    pst.control_data.noptmax = -1
    pst.model_command = 'python forward_pbrun.py'
    pst.pestpp_options['mou_dv_population_file'] = 'merged.dv_pop.csv'
    pst.pestpp_options['mou_obs_population_restart_file'] = 'merged.obs_pop.csv'
    pst.write(os.path.join('template_repo_update', "outer_repo.pst"))

    print(f"\n{datetime.datetime.now()}: outer repo update template prepped \n")

    #prep inner iter pst file
    print(f"\n{datetime.datetime.now()}: prepping inner template \n")
    pst = pyemu.Pst(pst_files[0])
    if os.path.exists('template_inner'):
        shutil.rmtree('template_inner')
    shutil.copytree('template', 'template_inner')

    pst.control_data.noptmax = nmax_inner
    pst.model_command = 'python forward_gprun.py'
    pst.pestpp_options['mou_objectives'] = 'func, func_var'
    pst.pestpp_options['mou_save_population_every'] = 1
    pst.pestpp_options['mou_ppd_beta'] = 0.7
    pst.pestpp_options['mou_env_selector'] = 'NSGA_PPD'
    pst.pestpp_options['mou_population_size'] = pop_size
    pst.pestpp_options['mou_max_archive_size'] = 100
    pst.pestpp_options['mou_dv_population_file'] = 'initial.dv_pop.csv'
    pst.pestpp_options['mou_obs_population_restart_file'] = 'initial.obs_pop.csv'
    pst.write(os.path.join('template_inner', os.path.basename(pst_files[0])))

    print(f"\n{datetime.datetime.now()}: inner template prepped \n")

def parse_all_io(inner_dirs):
    csvfiles = sorted(glob.glob(f"{inner_dirs[-1]}/*[0-999].dv_pop.csv", recursive=True), 
                      key=lambda x: int(x.split(".dv")[0].split(".")[1]))
    all_dv_list = []
    for file in csvfiles:
        generation = int(file.split(".dv")[0].split(".")[1])
        df = pd.read_csv(file).assign(generation=generation)
        df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
        all_dv_list.append(df)
    all_dv = pd.concat(all_dv_list, ignore_index=True)
    # all_dv.to_csv(os.path.join(inner_dirs[-1], "dv.summary.csv"), index=False)
    all_dv.drop(columns=['generation'], inplace=True)

    csvfiles = sorted(glob.glob(f"{inner_dirs[-1]}/*[0-999].obs_pop.csv", recursive=True), 
                      key=lambda x: int(x.split(".obs")[0].split(".")[1]))
    all_obs_list = []
    for file in csvfiles:
        generation = int(file.split(".obs")[0].split(".")[1])
        df = pd.read_csv(file).assign(generation=generation)
        df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
        all_obs_list.append(df)
    all_obs = pd.concat(all_obs_list, ignore_index=True)
    all_obs.to_csv(os.path.join(inner_dirs[-1], "obs.summary.csv"), index=False)
    all_obs.drop(columns=['generation'], inplace=True)

    return all_dv, all_obs

def resample(inner_dirs, outer_dirs):
    #get current training dv and obs dataset
    training_dv = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/gp_0.dv_training.csv", recursive=True)[0])
   
    #get all dv and obs visited in inner iters
    all_dv, all_obs = parse_all_io(inner_dirs)
    inner_pareto_summary = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/*.pareto.summary.csv", recursive=True)[0])

    dv_parname = [col for col in all_dv.columns if col in training_dv.columns and col != 'real_name']
    duplicates = pd.merge(all_dv[['real_name'] + dv_parname], training_dv[dv_parname], on=dv_parname, how='inner')
    inner_pareto = inner_pareto_summary[~inner_pareto_summary['member'].isin(duplicates['real_name'])]

    n_infill = 0
    iter_n = max(inner_pareto_summary['generation'])
    infill_pool = pd.DataFrame(columns = inner_pareto_summary.columns.values)

    while n_infill < max_infill and iter_n >= 0:
        infill_sort_gen = inner_pareto[(inner_pareto["generation"] == iter_n) & (~inner_pareto['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
        max_front_idx = max(infill_sort_gen['nsga2_front'])

        front_idx = 1

        while n_infill < max_infill and front_idx <= max_front_idx:
            infill_sort_front = infill_sort_gen[(infill_sort_gen['nsga2_front'] == front_idx) & (~infill_sort_gen['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
            size_to_fill = max_infill - n_infill
            
            if not infill_sort_front.empty:
                if size_to_fill < infill_sort_front.shape[0]:
                    infill_sort_front = infill_sort_front.sort_values(by='nsga2_crowding_distance', ascending=False)
                    infill_sort_front = infill_sort_front.head(size_to_fill)
                infill_pool = pd.concat([infill_pool, infill_sort_front], ignore_index=True)
                n_infill = infill_pool.shape[0]
            else:
                front_idx += 1 
               
        iter_n -= 1

    if infill_pool.shape[0] < max_infill:
        print(f"\n{datetime.datetime.now()}: WARNING: not enough infill points to fill pool. Continuing... \n")
    
    infill_pool_dv = all_dv[all_dv['real_name'].isin(infill_pool['member'].values)]
    infill_pool_dv.to_csv(os.path.join("template_outer", "infill.dv_pop.csv"), index=False)
    
    print(f"\n{datetime.datetime.now()}: infill ensemble saved \n")

    #sampling from decision space for restart dv file
    augmented_training_dv = pd.concat([training_dv, infill_pool_dv], ignore_index=True)
    kmeans = KMeans(n_clusters=pop_size).fit(augmented_training_dv.drop(columns='real_name'))
    restart_pool_dv = pd.DataFrame(kmeans.cluster_centers_, columns=training_dv.drop(columns='real_name').columns)
    restart_pool_dv.insert(0, 'real_name', [f'gen={max(inner_pareto["generation"]) * len(inner_dirs)}_restart={i+1}' for i in range(len(restart_pool_dv))])    
    restart_pool_dv.to_csv(os.path.join(tmpl_in, "initial.dv_pop.csv"), index=False)

    print(f"\n{datetime.datetime.now()}: restart population saved \n")
    

if __name__ == "__main__":

    print('Saving outputs to:', output_dir)

    start_time = datetime.datetime.now()
    print(f"\n{datetime.datetime.now()}: starting BBGO run \n")
    
    inner_dirs, outer_dirs = get_dirlist()

    if not restart:
        prep_templates()
        for d in inner_dirs + outer_dirs:
            shutil.rmtree(d)
        inner_dirs, outer_dirs = get_dirlist()

    for i in range(nmax_outer+1):
        if not outer_dirs:
            outer_dirs = outer_sweep(0)
            print(f"\n{datetime.datetime.now()}: outer 0 done \n")
        else:
            if not (restart and i == 0):
                outer_dirs = outer_sweep(int(outer_dirs[-1].split("_")[1]) + 1)
                update_outer_repo(outer_dirs)
        

        if i == nmax_outer:
            print(f"\n{datetime.datetime.now()}: outer {nmax_outer} done \n")
            print(f"total run time: {datetime.datetime.now() - start_time} \n")
            break

        inner_prep(inner_dirs, outer_dirs)
        next_inner_index = 1 if len(inner_dirs) == 0 else int(inner_dirs[-1].split("_")[1]) + 1
        inner_dirs = inner_opt(next_inner_index)
            
        resample(inner_dirs, outer_dirs)





