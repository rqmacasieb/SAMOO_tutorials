import os
import sys
import pyemu

def run():
    num_workers = 6
    tmpl_in = "kur_pbm_template"
    sys.path.insert(0,tmpl_in)
    from forward_pbrun import ppw_worker as ppw_function
    pyemu.os_utils.start_workers(tmpl_in, "pestpp-mou", "kur.pst", num_workers = num_workers,
                                worker_root = '.', master_dir = "pbm_run",
                                ppw_function = ppw_function)
    sys.path.remove(tmpl_in)

if __name__ == "__main__":
    run()
