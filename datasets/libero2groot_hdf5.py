from libero.libero import benchmark, get_libero_path
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import imageio
import os
from tqdm import tqdm

bddl_files_default_path = get_libero_path("bddl_files")



import h5py
from pathlib import Path
import json
from libero.libero.envs import OffScreenRenderEnv


demo_file = Path('/home/andang/workspace/mmint/neurips_2025/GROOT/datasets/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5')
bddl_path = str(Path(bddl_files_default_path) / 'libero_object' / (demo_file.stem[:-5] + '.bddl'))

# NOTE: go from one hdf5 file with multiple demos to multiple hdf5 files with one demo each
SAVE_DIR = "/home/andang/workspace/mmint/neurips_2025/GROOT/datasets/pick_up_the_orange_juice_and_place_it_in_the_basket_demo/"
os.makedirs(SAVE_DIR, exist_ok=True)

with h5py.File(demo_file, "r") as f:
    env_metadata = json.loads(f["data"].attrs["env_args"])
    data = f['data']
    
    demo_names = list(data.keys())
    for demo_name in tqdm(demo_names):
        # save file
        demo = data[demo_name]
        actions = demo['actions']
        observations = demo['obs']
        save_file = os.path.join(SAVE_DIR, f"{demo_name}.hdf5")
        print(f"Saving {save_file}")
        
        with h5py.File(save_file, "w") as f_save:
            data_grp = f_save.create_group("data")
            demo_grp = data_grp.create_group("demo_0")
            obs_grp  = demo_grp.create_group("obs")
            
            f_save["data"].attrs["bddl_path"] = bddl_path
            f_save["data"].attrs["demo_name"] = demo_name
            f_save["data"].attrs["env_args"] = f["data"].attrs["env_args"]
            
            demo_grp.create_dataset(f"actions", data=actions)
            for key, value in observations.items():
                obs_grp.create_dataset(key, data=value)