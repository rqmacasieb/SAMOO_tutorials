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

class SAMOO:
    """
    Surrogate-Assisted Multi-Objective Optimization (SAMOO) utility class.
    Manages the workflow for surrogate-based optimization with inner and outer loops.
    """
    def __init__(self, 
                 output_dir='.',
                 port=4000,
                 nmax_outer=20,
                 nmax_inner=50,
                 restart=False,
                 num_workers=8,
                 max_infill=100,
                 pop_size=100,
                 tmpl_in="template_inner"):
        """
        Initialize the SAMOO optimization framework.
        
        Parameters:
        -----------
        output_dir : str
            Directory for output files
        port : int
            Port number for worker communication
        nmax_outer : int
            Maximum number of outer iterations
        nmax_inner : int
            Maximum number of inner iterations
        restart : bool
            Whether to continue from last outer iteration
        num_workers : int
            Number of parallel workers
        max_infill : int
            Maximum number of infill points
        pop_size : int
            Population size for optimization
        tmpl_in : str
            Template directory for inner loop
        """
        self.output_dir = output_dir
        self.port = port
        self.nmax_outer = nmax_outer
        self.nmax_inner = nmax_inner
        self.restart = restart
        self.num_workers = num_workers
        self.max_infill = max_infill
        self.pop_size = pop_size
        self.tmpl_in = tmpl_in
        
    def inner_opt(self, iitidx):
        """Run inner optimization loop."""
        sys.path.insert(0, self.tmpl_in)
        from forward_gprun import ppw_worker as ppw_function 
        pyemu.os_utils.start_workers(self.tmpl_in, "pestpp-mou", 
                                    "ros_bbgo.pst", num_workers=self.num_workers, 
                                    worker_root=".", master_dir=f"./inner_{iitidx}", port=self.port,
                                    ppw_function=ppw_function)
        sys.path.remove(self.tmpl_in)

        # Delete some files to save space
        file_formats_to_delete = ["*.gp", "*.trimmed.archive.summary.csv"]
        for file_format in file_formats_to_delete:
            files_to_delete = glob.glob(os.path.join(f"./inner_{iitidx}", file_format))
            for file in files_to_delete:
                if os.path.exists(file):
                    os.remove(file)

        return self.get_inner_dirs()
        
    def outer_sweep(self, oitidx):
        """Run outer optimization loop."""
        if oitidx == 0:
            shutil.copy(os.path.join("template_outer", "ros_gp.lhs.dv_pop.csv"), 
                        os.path.join("template_outer", "infill.dv_pop.csv"))

        sys.path.insert(0, "template_outer")
        from forward_pbrun import ppw_worker as ppw_function 
        pyemu.os_utils.start_workers("template_outer", "pestpp-mou", 
                                    "ros_bbgo.pst", num_workers=self.num_workers, 
                                    worker_root=".", master_dir=f"./outer_{oitidx}", port=self.port,
                                    ppw_function=ppw_function)
        sys.path.remove("template_outer")

        return self.get_outer_dirs()

    def get_inner_dirs(self):
        """Get sorted list of inner directories."""
        return sorted([d for d in os.listdir() if d.startswith("inner_") and os.path.isdir(d)], 
                     key=lambda x: int(x.split("_")[1]))
    
    def get_outer_dirs(self):
        """Get sorted list of outer directories."""
        return sorted([d for d in os.listdir() if d.startswith("outer_") and os.path.isdir(d)], 
                     key=lambda x: int(x.split("_")[1]))

    def get_dirlist(self):
        """Get sorted lists of inner and outer directories."""
        return self.get_inner_dirs(), self.get_outer_dirs()

    def build_training_dataset(self, X, Y):
        """Build and save training dataset for GP model."""
        X.to_csv(os.path.join(self.tmpl_in, "gp_0.dv_training.csv"), index=False)
        X_values = X.drop(columns=['real_name']).values

        Y.to_csv(os.path.join(self.tmpl_in, "gp_0.obs_training.csv"), index=False)
        Y_values = Y.drop(columns=['real_name'])['func'].values

        # gpr.buildGP(X_values, Y_values, fname=os.path.join(self.tmpl_in, "gp_0.gp"))
        print(f"\n{datetime.datetime.now()}: GP training dataset saved. \n")

    def inner_prep(self, inner_dirs, outer_dirs):
        """Prepare for inner optimization loop."""
        curr_dv = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/*0.dv_pop.csv", recursive=True)[0])
        curr_obs = pd.read_csv(glob.glob(f"./{outer_dirs[-1]}/*0.obs_pop.csv", recursive=True)[0])

        # Copy curr opt for ei calcs
        if outer_dirs[-1].endswith("_0"):
            curr_opt_obs = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "*0.archive.obs_pop.csv"), recursive=True)[0])
            curr_opt_dv = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "*0.archive.dv_pop.csv"), recursive=True)[0])
        else:
            curr_opt_obs = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "outer_repo", "*.archive.obs_pop.csv"), recursive=True)[0])
            curr_opt_dv = pd.read_csv(glob.glob(os.path.join(outer_dirs[-1], "outer_repo", "*.archive.dv_pop.csv"), recursive=True)[0])
        
        curr_opt_obs.to_csv(os.path.join(self.tmpl_in, "curr_opt.csv"), index=False)

        # Copy previous outer repo update dv and obs files
        if outer_dirs[-1].endswith("_0"):      
            restart_dv = pd.read_csv(os.path.join(self.tmpl_in, "starter.dv_pop.csv"))
        else:
            restart_dv = pd.read_csv(os.path.join(self.tmpl_in, "initial.dv_pop.csv"))

        restart_dv = restart_dv.loc[~restart_dv.duplicated(subset=restart_dv.columns.difference(['real_name']), keep='first')]
        restart_dv.to_csv(os.path.join(self.tmpl_in, "initial.dv_pop.csv"), index=False)

        # Remove existing gp files and training data from template dir
        for file in glob.glob(os.path.join(self.tmpl_in, "*.gp")) + \
                    glob.glob(os.path.join(self.tmpl_in, "*dv_training.csv")) + \
                    glob.glob(os.path.join(self.tmpl_in, "*obs_training.csv")):
            os.remove(file)

        # Update training data
        if outer_dirs[-1].endswith("_0"):
            training_dv, training_obs = curr_dv, curr_obs
        else:
            training_dv = pd.concat([pd.read_csv(glob.glob(os.path.join(inner_dirs[-1], "*0.dv_training.csv"), recursive=True)[0]), 
                                    curr_dv], ignore_index=True)
            training_obs = pd.concat([pd.read_csv(glob.glob(os.path.join(inner_dirs[-1], "*0.obs_training.csv"), recursive=True)[0]), 
                                    curr_obs], ignore_index=True)
            training_dv.columns, training_obs.columns = curr_dv.columns.values, curr_obs.columns.values

        self.build_training_dataset(training_dv, training_obs)

    def update_outer_repo(self, outer_dirs):
        """Update outer repository with new solutions."""
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

        # Run pestpp mou in pareto sorting mode
        pyemu.os_utils.start_workers("template_repo_update", "pestpp-mou", 
                                    "outer_repo.pst", num_workers=self.num_workers, port=self.port,
                                    worker_root=".", master_dir="temp")

        # Copy outer repo update files to outer dir
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
        
        # Clean up temp directory
        shutil.rmtree("temp")

    def prep_templates(self):
        """Prepare templates for optimization."""
        print(f"\n{datetime.datetime.now()}: prepping templates \n")
              
        pst_files = glob.glob(os.path.join('template', '*.pst'))
        if len(pst_files) != 1:
            raise ValueError("There should be exactly one .pst file in the template directory.")
        
        # Prep outer iter pst file
        print(f"\n{datetime.datetime.now()}: prepping outer template \n")
        pst = pyemu.Pst(pst_files[0])
        if os.path.exists('template_outer'):
            shutil.rmtree('template_outer')
        shutil.copytree('template', 'template_outer')

        pst.model_command = 'python forward_pbrun.py'
        pst.write(os.path.join('template_outer', os.path.basename(pst_files[0])))

        print(f"\n{datetime.datetime.now()}: outer template prepped \n")

        # Prep outer repo update template
        print(f"\n{datetime.datetime.now()}: prepping repo update template \n")
        pst = pyemu.Pst(pst_files[0])
        if os.path.exists('template_repo_update'):
            shutil.rmtree('template_repo_update')
        shutil.copytree('template', 'template_repo_update')

        pst.model_command = 'python forward_pbrun.py'
        pst.pestpp_options['mou_dv_population_file'] = 'merged.dv_pop.csv'
        pst.pestpp_options['mou_obs_population_restart_file'] = 'merged.obs_pop.csv'
        pst.write(os.path.join('template_repo_update', "outer_repo.pst"))

        print(f"\n{datetime.datetime.now()}: repo update template prepped \n")

        # Prep inner iter pst file
        print(f"\n{datetime.datetime.now()}: prepping inner template \n")
        pst = pyemu.Pst(pst_files[0])
        if os.path.exists('template_inner'):
            shutil.rmtree('template_inner')
        shutil.copytree('template', 'template_inner')

        pst.control_data.noptmax = self.nmax_inner
        obs = pst.observation_data
        obs.loc['func_var', 'obgnme'] = 'l_obj'
        pst.model_command = 'python forward_gprun.py'
        pst.pestpp_options['mou_objectives'] = 'func, func_var'
        pst.pestpp_options['mou_save_population_every'] = 1
        pst.pestpp_options['mou_population_size'] = self.pop_size
        pst.pestpp_options['mou_max_archive_size'] = 100
        pst.pestpp_options['mou_dv_population_file'] = 'initial.dv_pop.csv'
        pst.write(os.path.join('template_inner', os.path.basename(pst_files[0])))

        print(f"\n{datetime.datetime.now()}: inner template prepped \n")

    def modify_inner_pst(self, inner_dirs):
        """Modify inner PST file based on optimization progress."""
        pst_files = glob.glob(os.path.join('template_inner', '*.pst'))
        pst = pyemu.Pst(pst_files[0])
        obs = pst.observation_data
        current_direction = 'max' if obs.loc['func_var', 'obgnme'] == 'g_obj' else 'min'

        if current_direction == 'max':
            new_direction = 'min'
        else:
            prev_inner_pareto = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/*.pareto.summary.csv", recursive=True)[0])
            prev_inner_pareto = prev_inner_pareto[(prev_inner_pareto['generation'] == prev_inner_pareto['generation'].max()) & (prev_inner_pareto['nsga2_front'] == 1)]
            # Check if we need to modify the inner PST file based on pareto front size
            if len(prev_inner_pareto) < self.max_infill / 2:
                print(f"\n{datetime.datetime.now()}: Pareto front size ({len(prev_inner_pareto)}) is less than half of max_infill ({self.max_infill/2})")
                print(f"\n{datetime.datetime.now()}: Modifying inner PST to switch optimization direction\n")
                new_direction = 'max'

                prev_obs_summary = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/obs.summary.csv", recursive=True)[0])
                sd_cutoff = np.sqrt(prev_obs_summary['func_var'].quantile(0.90))
            else:
                new_direction = 'min'

        if new_direction == 'max':
            obs.loc['func_var', 'obgnme'] = 'g_obj'
            obs.loc['func_sd', 'obgnme'] = 'l_than'
            obs.loc['func_sd', 'obsval'] = sd_cutoff
            pst.pestpp_options['opt_constraint_groups'] = 'l_than, g_than'
        else:
            obs.loc['func_var', 'obgnme'] = 'l_obj'
            obs.loc['func_sd', 'obgnme'] = 'obs'
            pst.pestpp_options['opt_constraint_groups'] = 'g_than'

        pst.write(os.path.join('template_inner', os.path.basename(pst_files[0])))

    def parse_all_io(self, inner_dirs):
        """Parse all input/output data from inner iterations."""
        csvfiles = sorted(glob.glob(f"{inner_dirs[-1]}/*[0-999].dv_pop.csv", recursive=True), 
                        key=lambda x: int(x.split(".dv")[0].split(".")[1]))
        all_dv_list = []
        for file in csvfiles:
            generation = int(file.split(".dv")[0].split(".")[1])
            df = pd.read_csv(file).assign(generation=generation)
            df = df[['generation'] + [col for col in df.columns if col != 'generation']] 
            all_dv_list.append(df)
        all_dv = pd.concat(all_dv_list, ignore_index=True)
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

    def resample(self, inner_dirs, outer_dirs):
        """Resample points for next iteration."""
        # Get current training dv and obs dataset
        training_dv = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/gp_0.dv_training.csv", recursive=True)[0])
       
        # Get all dv and obs visited in inner iters
        all_dv, all_obs = self.parse_all_io(inner_dirs)
        inner_pareto_summary = pd.read_csv(glob.glob(f"{inner_dirs[-1]}/*.pareto.summary.csv", recursive=True)[0])

        dv_parname = [col for col in all_dv.columns if col in training_dv.columns and col != 'real_name']
        duplicates = pd.merge(all_dv[['real_name'] + dv_parname], training_dv[dv_parname], on=dv_parname, how='inner')
        inner_pareto = inner_pareto_summary[~inner_pareto_summary['member'].isin(duplicates['real_name'])]

        n_infill = 0
        iter_n = max(inner_pareto_summary['generation'])
        infill_pool = pd.DataFrame(columns = inner_pareto_summary.columns.values)

        while n_infill < self.max_infill and iter_n >= 0:
            infill_sort_gen = inner_pareto[(inner_pareto["generation"] == iter_n) & (~inner_pareto['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
            max_front_idx = max(infill_sort_gen['nsga2_front']) if not infill_sort_gen.empty else 0

            front_idx = 1

            while n_infill < self.max_infill and front_idx <= max_front_idx:
                infill_sort_front = infill_sort_gen[(infill_sort_gen['nsga2_front'] == front_idx) & (~infill_sort_gen['member'].isin(infill_pool['member']))].drop_duplicates(subset='member')
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

        # Sampling from decision space for restart dv file
        augmented_training_dv = pd.concat([training_dv, infill_pool_dv], ignore_index=True)
        kmeans = KMeans(n_clusters=self.pop_size).fit(augmented_training_dv.drop(columns='real_name'))
        restart_pool_dv = pd.DataFrame(kmeans.cluster_centers_, columns=training_dv.drop(columns='real_name').columns)
        restart_pool_dv.insert(0, 'real_name', [f'gen={max(inner_pareto["generation"]) * len(inner_dirs)}_restart={i+1}' for i in range(len(restart_pool_dv))])    
        restart_pool_dv.to_csv(os.path.join(self.tmpl_in, "initial.dv_pop.csv"), index=False)

        print(f"\n{datetime.datetime.now()}: restart population saved \n")
    
    def run(self):
        """Run the full SAMOO optimization workflow."""
        print(f'Saving outputs to: {self.output_dir}')

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


# For backward compatibility
def inner_opt(iitidx):
    samoo = SAMOO()
    return samoo.inner_opt(iitidx)

def outer_sweep(oitidx):
    samoo = SAMOO()
    return samoo.outer_sweep(oitidx)

def get_dirlist():
    samoo = SAMOO()
    return samoo.get_dirlist()

def build_training_dataset(X, Y):
    samoo = SAMOO()
    return samoo.build_training_dataset(X, Y)

def inner_prep(inner_dirs, outer_dirs):
    samoo = SAMOO()
    return samoo.inner_prep(inner_dirs, outer_dirs)

def update_outer_repo(outer_dirs):
    samoo = SAMOO()
    return samoo.update_outer_repo(outer_dirs)

def prep_templates():
    samoo = SAMOO()
    return samoo.prep_templates()

def modify_inner_pst(inner_dirs):
    samoo = SAMOO()
    return samoo.modify_inner_pst(inner_dirs)

def parse_all_io(inner_dirs):
    samoo = SAMOO()
    return samoo.parse_all_io(inner_dirs)

def resample(inner_dirs, outer_dirs):
    samoo = SAMOO()
    return samoo.resample(inner_dirs, outer_dirs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='.')
    parser.add_argument('--port', type=int, default=4000)
    parser.add_argument('--nmax-outer', type=int, default=20)
    parser.add_argument('--nmax-inner', type=int, default=50)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-infill', type=int, default=100)
    parser.add_argument('--pop-size', type=int, default=100)
    args = parser.parse_args()
    
    samoo = SAMOO(
        output_dir=args.output_dir,
        port=args.port,
        nmax_outer=args.nmax_outer,
        nmax_inner=args.nmax_inner,
        restart=args.restart,
        num_workers=args.num_workers,
        max_infill=args.max_infill,
        pop_size=args.pop_size
    )
    
    samoo.run()