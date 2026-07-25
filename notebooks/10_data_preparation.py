# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: project-thesis (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Preparation

# %% [markdown]
# ## Dataset: Teeth3DS+
#
# Download the following archives from: https://osf.io/xctdy
#
#   - 3DTeethSeg22_challenge_train_test_split.zip
#   - Teeth3DS_train_test_split.zip
#   - license.txt
#   - data_part_1.zip
#   - ...
#   - data_part_7.zip

# %%
import sys
from pathlib import Path

import cv2
import torch
import yaml
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src import file_io, preprocessing

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
dataset_path_2d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_2d={dataset_path_2d}")

dataset_path_3d = root_path / config.get("dataset_path_3d")
dataset_path_3d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_3d={dataset_path_3d}")

# %%
# load Teeth3DS+ dataset
# https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Teeth3DS.html#torch_geometric.datasets.Teeth3DS

from torch_geometric.datasets import Teeth3DS

split = "3DTeethSeg22_challenge",  # Teeth3DS, 3DTeethSeg22_challenge, 3DTeethLand_challenge

train_dataset = Teeth3DS(
    root=dataset_path_3d,
    split=split,
    train=True,
)

test_dataset = Teeth3DS(
    root=dataset_path_3d,
    split=split,
    train=False,
)

# %%
# dataset still contains samples from 3DTeethLand_challenge (no vertex label files available), remove those

import shutil

obj_files = list(dataset_path_3d.rglob("*.obj"))
for obj_file in obj_files:
    vertex_label_file = obj_file.with_suffix(".json")
    if not vertex_label_file.exists():
        print(f"removing sample: '{vertex_label_file.parent}'")
        shutil.rmtree(obj_file.parent, ignore_errors=True)

# %% [markdown]
# ## Dataset: Teeth2D
#
# Create the image dataset `Teeth2D` by rendering multiple views of the 3D mesh models. 

# %%
images_path = dataset_path_2d / "images"
images_path.mkdir(parents=True, exist_ok=True)
print(f"images_path={images_path}")

masks_path = dataset_path_2d / "masks"
masks_path.mkdir(parents=True, exist_ok=True)
print(f"masks_path={masks_path}")

preproc = preprocessing.Preprocessing(config, device)

obj_files = list(dataset_path_3d.rglob("*.obj"))
print(f"number of scans: {len(obj_files)}")

for obj_file in tqdm(obj_files, desc="Rendering 2D views"):
    mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)
    vertex_labels = file_io.load_vertex_labels(obj_file.with_suffix(".json"))
    images, masks = preproc.render_2d_views(mesh, vertex_labels)

    for image, mask, view in zip(images, masks, preproc.views):
        elev = view[0]
        azim = view[1]

        image_file = images_path / obj_file.stem / f"image_elev{elev}_azim{azim}.png"
        image_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_file, image)

        mask_file = masks_path / obj_file.stem / f"mask_elev{elev}_azim{azim}.png"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(mask_file, mask)
