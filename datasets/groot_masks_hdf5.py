import argparse
import sys
import os

import h5py
import cv2
import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

project_repo_folder = "."
# For XMem
sys.path.append(f"{project_repo_folder}/third_party/XMem")
sys.path.append(f"{project_repo_folder}/third_party/XMem/model")
sys.path.append(f"{project_repo_folder}/third_party/XMem/util")
sys.path.append(f"{project_repo_folder}/third_party/XMem/inference")
sys.path.append(f"{project_repo_folder}/")
from PIL import Image
from pathlib import Path
from groot_imitation.groot_algo import GROOT_ROOT_PATH
from groot_imitation.groot_algo.xmem_tracker import XMemTracker
from groot_imitation.groot_algo.misc_utils import get_annotation_path, get_first_frame_annotation, overlay_xmem_mask_on_image, depth_to_rgb, resize_image_to_same_shape, plotly_draw_seg_image, rotate_camera_pose
from groot_imitation.groot_algo.misc_utils import overlay_xmem_mask_on_image, add_palette_on_mask, VideoWriter, get_transformed_depth_img
from groot_imitation.groot_algo.o3d_modules import O3DPointCloud, convert_convention
from tqdm import tqdm

def insert_masks_into_hdf5(hdf5_file, annotation_folder):
    
    # check if video_masks.hdf5 already exists as indicator that we should skip
    if os.path.exists(os.path.join(annotation_folder, "video_masks.hdf5")):
        return # skip this demo
    
    first_frame, first_frame_annotation = get_first_frame_annotation(annotation_folder)

    # ************************ Most important part *******************************
    xmem_tracker = XMemTracker(xmem_checkpoint=f'{project_repo_folder}/third_party/xmem_checkpoints/XMem.pth', device='cuda:0')
    xmem_tracker.clear_memory()
    # **************************************************************************

    resized_images = []

    with h5py.File(hdf5_file, "r") as f:
        images = f["data/demo_0/obs"]["agentview_rgb"][:]

    for image in images:
        image = cv2.resize(image, (first_frame_annotation.shape[1], first_frame_annotation.shape[0]), interpolation=cv2.INTER_AREA)
        resized_images.append(image)

    masks = xmem_tracker.track_video(resized_images, first_frame_annotation)


    mask_file = os.path.join(annotation_folder, "video_masks.hdf5")

    with h5py.File(hdf5_file, "r+") as f: # add agentview_masks
        f["data/demo_0/obs"].create_dataset("agentview_masks", data=np.stack(masks, axis=0))

    with h5py.File(mask_file, "w") as f:
        f.create_group("data")
        f["data"].create_dataset("agentview_masks", data=np.stack(masks, axis=0))

    with VideoWriter(video_path=annotation_folder, video_name="mask_only_video.mp4", fps=20, save_video=True) as video_writer:
        for mask, image in zip(masks, resized_images):
            new_mask_img = add_palette_on_mask(mask).convert("RGB")
            video_writer.append_image(np.array(new_mask_img))

    with VideoWriter(video_path=annotation_folder, video_name="overlay_video.mp4", fps=20, save_video=True) as video_writer:
        for mask, image in zip(masks, resized_images):
            new_mask_img = overlay_xmem_mask_on_image(image, mask, use_white_bg=True)
            video_writer.append_image(np.array(new_mask_img))
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert GROOT dataset to HDF5 format.")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the input GROOT dataset directory.', default=f"{project_repo_folder}/datasets/pick_up_the_orange_juice_and_place_it_in_the_basket_demo/")
    parser.add_argument('--annot_dir', type=str, required=True, help='Path to the input annotation directory.', default=f"{project_repo_folder}/datasets/annotations/")
    args = parser.parse_args()
    
    demo_files = [os.path.join(args.dataset_dir, demo_filename) for demo_filename in os.listdir(args.dataset_dir)]
    annotation_folders = [os.path.join(args.annot_dir, demo_filename.split('.')[0]) for demo_filename in os.listdir(args.dataset_dir)]
    
    # annotation_folder = f"{project_repo_folder}/datasets/annotations/demo_0"
    # demo_file_name = f"{project_repo_folder}/datasets/pick_up_the_orange_juice_and_place_it_in_the_basket_demo/demo_0.hdf5"
    
    
    for demo_file, annotation_folder in tqdm(zip(demo_files, annotation_folders)):
        # print(f"Processing {demo_file} and {annotation_folder}")
        insert_masks_into_hdf5(demo_file, annotation_folder)