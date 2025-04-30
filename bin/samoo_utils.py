import pandas as pd
import os
import laGPy as gpr
import pyemu
import glob
import sys
import shutil
import datetime
import numpy as np
import argparse
import re

class SAMOO:
    def __init__(self, 
                 nmax_outer=1, 
                 nmax_inner=10, 
                 restart=False, 
                 num_workers=8, 
                 max_infill=20, 
                 repo_size = 30,
                 pop_size=30, 
                 ppd_beta = 0.6,
                 tmpl_in="template_inner",
                 exe_file = './pestpp-mou',
                 output_dir='.',
                 port=4000):
        
        self.nmax_outer = nmax_outer
        self.nmax_inner = nmax_inner
        self.restart = restart
        self.num_workers = num_workers
        self.max_infill = max_infill
        self.pop_size = pop_size
        self.repo_size = repo_size
        self.ppd_beta = ppd_beta
        self.tmpl_in = tmpl_in
        self.output_dir = output_dir
        self.exe_file = exe_file
        self.port = port
        
    def inner_opt(self, iitidx):
        pst_file = os.path.basename(glob.glob(os.path.join(self.tmpl_in, '*.pst'))[0])   
        sys.path.insert(0, self.tmpl_in)
        from forward_gprun import ppw_worker as ppw_function 
        pyemu.os_utils.start_workers(self.tmpl_in, self.exe_file, 
                                    pst_file, num_workers=self.num_workers, 
                                    worker_root=".", master_dir=f"./inner_{iitidx}", port=self.port,
                                    ppw_function=ppw_function)
        sys.path.remove(self.tmpl_in)

        #delete some files to save space
        file_formats_to_delete = ["*.gp", "*.trimmed.archive.summary.csv"]
        for file_format in file_formats_to_delete:
            files_to_delete = glob.glob(os.path.join(f"./inner_{iitidx}", file_format))
            for file in files_to_delete:
                if os.path.exists(file):
                    os.remove(file)

        return self.get_inner_dirs()
        
    def outer_sweep(self, oitidx):
        pst_file = os.path.basename(glob.glob(os.path.join('template_outer', '*.pst'))[0])   
        if oitidx == 0:
            shutil.copy(os.path.join("template_outer", "gp.lhs.dv_pop.csv"), 
                        os.path.join("template_outer", "infill.dv_pop.csv"))

        sys.path.insert(0, "template_outer")
        from forward_pbrun import ppw_worker as ppw_function 
        pyemu.os_utils.start_workers("template_outer", self.exe_file, 
                                    pst_file, num_workers=self.num_workers, 
                                    worker_root=".", master_dir=f"./outer_{oitidx}", port=self.port,
                                    ppw_function=ppw_function)
        sys.path.remove("template_outer")

        return self.get_outer_dirs()

    def get_dirlist(self):
        inner_dirs = self.get_inner_dirs()
        outer_dirs = self.get_outer_dirs()
        return inner_dirs, outer_dirs
    
    def get_inner_dirs(self):
        return sorted([d for d in os.listdir() if d.startswith("inner_") and os.path.isdir(d)], 
                      key=lambda x: int(x.split("_")[1]))
        
    def get_outer_dirs(self):
        return sorted([d for d in os.listdir() if d.startswith("outer_") and os.path.isdir(d)], 
                      key=lambda x: int(x.split("_")[1]))

    def inner_prep(self, inner_dirs, outer_dirs):
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
            if restart_dv.shape[0] < self.pop_size:
                if self.pop_size - restart_dv.shape[0] <= dv_restart_pool.shape[0]:
                    restart_dv_subsample = dv_restart_pool.sample(n=self.pop_size - restart_dv.shape[0])
                else:
                    restart_dv_subsample = dv_restart_pool

                restart_obs_subsample = curr_obs[curr_obs['real_name'].isin(restart_dv_subsample['real_name'])]
                restart_dv = pd.concat([restart_dv, restart_dv_subsample], ignore_index=True)
                restart_obs = pd.concat([restart_obs, restart_obs_subsample], ignore_index=True)
            else:
                restart_dv = restart_dv.sample(n=self.pop_size)
                restart_obs = restart_obs[restart_obs['real_name'].isin(restart_dv['real_name'])]

        restart_dv = restart_dv.loc[~restart_dv.duplicated(subset=restart_dv.columns.difference(['real_name']), keep='first')]
        restart_dv = restart_dv.sort_values('real_name')
        restart_dv.to_csv(os.path.join(self.tmpl_in, "initial.dv_pop.csv"), index=False)

        restart_obs = restart_obs.loc[~restart_obs.duplicated(subset=restart_obs.columns.difference(['real_name']), keep='first')]
        restart_obs = restart_obs.sort_values('real_name')
        restart_obs.to_csv(os.path.join(self.tmpl_in, "initial.obs_pop.csv"), index=False)

        #remove existing gp files and training data from template dir
        for file in glob.glob(os.path.join(self.tmpl_in, "*.gp")) + \
                    glob.glob(os.path.join(self.tmpl_in, "*dv_training.csv")) + \
                    glob.glob(os.path.join(self.tmpl_in, "*obs_training.csv")):
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

        training_dv.to_csv(os.path.join(self.tmpl_in, f"gp_0.dv_training.csv"), index=False)
        training_obs.to_csv(os.path.join(self.tmpl_in, f"gp_0.obs_training.csv"), index=False)

    def update_outer_repo(self, outer_dirs):
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
        pyemu.os_utils.start_workers("template_repo_update", self.exe_file, 
                                    "outer_repo.pst", num_workers=1, port=self.port,
                                    worker_root=".", master_dir="temp")

        # copy outer repo update files to outer dir
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

    def prep_templates(self):
        print(f"\n{datetime.datetime.now()}: prepping templates \n")
              
        pst_file = glob.glob(os.path.join('template', '*.pst'))
        if len(pst_file) != 1:
            raise ValueError("There should be exactly one .pst file in the template directory.")
        
        # prep outer iter pst file
        print(f"\n{datetime.datetime.now()}: prepping outer template \n")
        pst = pyemu.Pst(pst_file[0])
        if os.path.exists('template_outer'):
            shutil.rmtree('template_outer')
        shutil.copytree('template', 'template_outer')

        pst.control_data.noptmax = -1
        pst.model_command = 'python forward_pbrun.py'
        pst.pestpp_options['mou_dv_population_file'] = 'infill.dv_pop.csv'
        pst.write(os.path.join('template_outer', os.path.basename(pst_file[0])))

        print(f"\n{datetime.datetime.now()}: outer template prepped \n")

        # prep outer repo update template
        print(f"\n{datetime.datetime.now()}: prepping repo update template \n")
        pst = pyemu.Pst(pst_file[0])
        if os.path.exists('template_repo_update'):
            shutil.rmtree('template_repo_update')
        shutil.copytree('template', 'template_repo_update')

        pst.control_data.noptmax = -1
        pst.model_command = 'python forward_pbrun.py'
        pst.pestpp_options['mou_dv_population_file'] = 'merged.dv_pop.csv'
        pst.pestpp_options['mou_obs_population_restart_file'] = 'merged.obs_pop.csv'
        pst.write(os.path.join('template_repo_update', 'outer_repo.pst'))

        print(f"\n{datetime.datetime.now()}: outer repo update template prepped \n")

        #prep inner iter pst file
        print(f"\n{datetime.datetime.now()}: prepping inner template \n")
        pst = pyemu.Pst(pst_file[0])
        if os.path.exists('template_inner'):
            shutil.rmtree('template_inner')
        shutil.copytree('template', 'template_inner')

        pst.control_data.noptmax = self.nmax_inner
        pst.model_command = 'python forward_gprun.py'
        pst.pestpp_options['mou_save_population_every'] = 1
        pst.pestpp_options['mou_ppd_beta'] = self.ppd_beta
        pst.pestpp_options['mou_env_selector'] = 'NSGA_PPD'
        pst.pestpp_options['mou_population_size'] = self.pop_size
        pst.pestpp_options['mou_max_archive_size'] = self.repo_size
        pst.pestpp_options['mou_dv_population_file'] = 'initial.dv_pop.csv'
        pst.pestpp_options['mou_obs_population_restart_file'] = 'initial.obs_pop.csv'
        pst.write(os.path.join('template_inner', os.path.basename(pst_file[0])))

        print(f"\n{datetime.datetime.now()}: inner template prepped \n")

    def parse_all_io(self, inner_dir, save_consolidated_dv=False, save_consolidated_obs=False):
        csvfiles = sorted(glob.glob(f"{inner_dir}/*[0-999].dv_pop.csv", recursive=True), 
                        key=lambda x: int(x.split(".dv")[0].split(".")[1]))
        all_dv_list = []
        for file in csvfiles:
            generation = int(file.split(".dv")[0].split(".")[1])
            df = pd.read_csv(file).assign(generation=generation)
            df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
            all_dv_list.append(df)
        all_dv = pd.concat(all_dv_list, ignore_index=True)
        if save_consolidated_dv:
            all_dv.to_csv(os.path.join(inner_dir, "dv.summary.csv"), index=False)

        csvfiles = sorted(glob.glob(f"{inner_dir}/*[0-999].obs_pop.csv", recursive=True), 
                        key=lambda x: int(x.split(".obs")[0].split(".")[1]))
        all_obs_list = []
        for file in csvfiles:
            generation = int(file.split(".obs")[0].split(".")[1])
            df = pd.read_csv(file).assign(generation=generation)
            df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
            all_obs_list.append(df)
        all_obs = pd.concat(all_obs_list, ignore_index=True)
        if save_consolidated_obs:
            all_obs.to_csv(os.path.join(inner_dir, "obs.summary.csv"), index=False)

        return all_dv, all_obs

    def resample(self, inner_dirs, outer_dirs):
        #get current training dv and obs dataset
        training_dv = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/gp_0.dv_training.csv", recursive=True)[0])
       
        #get all dv and obs visited in inner iters
        all_dv, all_obs = self.parse_all_io(inner_dirs[-1])
        all_dv.drop(columns=['generation'], inplace=True)
        all_obs.drop(columns=['generation'], inplace=True)
        inner_pareto_summary = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/*.pareto.summary.csv", recursive=True)[0])

        dv_parname = [col for col in all_dv.columns if col in training_dv.columns and col != 'real_name']
        duplicates = pd.merge(all_dv[['real_name'] + dv_parname], training_dv[dv_parname], on=dv_parname, how='inner')
        inner_pareto = inner_pareto_summary[~inner_pareto_summary['member'].isin(duplicates['real_name'])]

        n_infill = 0
        iter_n = max(inner_pareto_summary['generation'])
        infill_pool = pd.DataFrame(columns=inner_pareto_summary.columns.values)

        while n_infill < self.max_infill and iter_n >= 0:
            infill_sort_gen = inner_pareto[(inner_pareto["generation"] == iter_n) & 
                                          (~inner_pareto['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
            if infill_sort_gen.empty:
                iter_n -= 1
                continue
                
            max_front_idx = max(infill_sort_gen['nsga2_front'])
            front_idx = 1

            while n_infill < self.max_infill and front_idx <= max_front_idx:
                infill_sort_front = infill_sort_gen[(infill_sort_gen['nsga2_front'] == front_idx) & 
                                                  (~infill_sort_gen['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
                size_to_fill = self.max_infill - n_infill
                
                if not infill_sort_front.empty:
                    if size_to_fill < infill_sort_front.shape[0]:
                        infill_sort_front = infill_sort_front.sort_values(by='nsga2_crowding_distance', ascending=False)
                        infill_sort_front = infill_sort_front.head(size_to_fill)
                    infill_pool = pd.concat([infill_pool, infill_sort_front], ignore_index=True)
                    n_infill = infill_pool.shape[0]
                else:
                    front_idx += 1 
            iter_n -= 1

        if infill_pool.shape[0] < self.max_infill:
            print(f"\n{datetime.datetime.now()}: WARNING: not enough infill points to fill pool. Continuing... \n")
        
        infill_pool_dv = all_dv[all_dv['real_name'].isin(infill_pool['member'].values)]
        infill_pool_dv.to_csv(os.path.join("template_outer", "infill.dv_pop.csv"), index=False)
        
        print(f"\n{datetime.datetime.now()}: infill ensemble saved \n")
    
    def run(self):
        print('Saving outputs to:', self.output_dir)

        start_time = datetime.datetime.now()
        print(f"\n{datetime.datetime.now()}: starting SAMOO run \n")
        
        inner_dirs, outer_dirs = self.get_dirlist()

        if not self.restart:
            self.prep_templates()
            for d in inner_dirs + outer_dirs:
                shutil.rmtree(d)
            inner_dirs, outer_dirs = self.get_dirlist()

        for i in range(self.nmax_outer+1):
            if not outer_dirs:
                outer_dirs = self.outer_sweep(0)
                print(f"\n{datetime.datetime.now()}: outer 0 done \n")
            else:
                if not (self.restart and i == 0):
                    outer_dirs = self.outer_sweep(int(outer_dirs[-1].split("_")[1]) + 1)
                    self.update_outer_repo(outer_dirs)

            if i == self.nmax_outer:
                print(f"\n{datetime.datetime.now()}: outer {self.nmax_outer} done \n")
                print(f"total run time: {datetime.datetime.now() - start_time} \n")
                break

            self.inner_prep(inner_dirs, outer_dirs)
            next_inner_index = 1 if len(inner_dirs) == 0 else int(inner_dirs[-1].split("_")[1]) + 1
            inner_dirs = self.inner_opt(next_inner_index)
                
            self.resample(inner_dirs, outer_dirs)

def parse_args():
    parser = argparse.ArgumentParser(description='SAMOO: Surrogate-Assisted Multi-Objective Optimization')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    parser.add_argument('--port', type=int, default=4000, help='Port number for workers')
    parser.add_argument('--nmax-outer', type=int, default=5, help='Maximum number of outer iterations')
    parser.add_argument('--nmax-inner', type=int, default=20, help='Maximum number of inner iterations')
    parser.add_argument('--restart', action='store_true', help='Restart from last outer iteration')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--max-infill', type=int, default=50, help='Maximum number of infill points')
    parser.add_argument('--pop-size', type=int, default=50, help='Population size')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    samoo = SAMOO(
        nmax_outer=args.nmax_outer,
        nmax_inner=args.nmax_inner,
        restart=args.restart,
        num_workers=args.num_workers,
        max_infill=args.max_infill,
        pop_size=args.pop_size,
        output_dir=args.output_dir,
        exe_file=args.exe_file,
        port=args.port        
    )
    
    samoo.run()