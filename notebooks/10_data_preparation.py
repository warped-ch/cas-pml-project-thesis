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

# %%
import shutil
import sys
from pathlib import Path

import cv2
import torch
import yaml
from torch_geometric.datasets import Teeth3DS
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src import file_io, view_projector

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_3d = root_path / config.get("dataset_path_3d")
dataset_path_3d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_3d={dataset_path_3d}")

projections_path = root_path / config["data_path_projections"]
projections_path.mkdir(parents=True, exist_ok=True)
print(f"projections_path={projections_path}")

# %% [markdown]
# ## Dataset: Teeth3DS+
#
# - Download [Teeth3DS](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Teeth3DS.html) dataset from PyTorch Geometric.
# - Clean up leftovers from `3DTeethLand_challenge` split.

# %%
split = (
    "3DTeethSeg22_challenge",
)  # Teeth3DS, 3DTeethSeg22_challenge, 3DTeethLand_challenge

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

# Teeth3DS+ dataset still contains samples from 3DTeethLand_challenge (no vertex label files available), remove those
obj_files = list(dataset_path_3d.rglob("*.obj"))
for obj_file in obj_files:
    vertex_label_file = obj_file.with_suffix(".json")
    if not vertex_label_file.exists():
        print(f"removing sample: '{vertex_label_file.parent}'")
        shutil.rmtree(obj_file.parent, ignore_errors=True)

# %% [markdown]
# ## Multi-View Projection Rendering
#
# - Render multi-view projections of the 3D models.
# - Save view and mask images.

# %%
shutil.rmtree(projections_path, ignore_errors=True)

images_path = projections_path / "images"
images_path.mkdir(parents=True, exist_ok=True)
print(f"images_path={images_path}")

masks_path = projections_path / "masks"
masks_path.mkdir(parents=True, exist_ok=True)
print(f"masks_path={masks_path}")

class_id_map = config["class_id_map"]

view_proj = view_projector.ViewProjector(config, device)

obj_files = list(dataset_path_3d.rglob("*.obj"))

for obj_file in tqdm(obj_files, desc="Rendering 2D views"):
    mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)
    vertex_labels = file_io.load_vertex_labels(
        obj_file.with_suffix(".json"), class_id_map
    )
    images, masks = view_proj.render_2d_views(mesh, vertex_labels)

    for image, mask, view in zip(images, masks, view_proj.views):
        elev = view[0]
        azim = view[1]

        image_file = images_path / f"{obj_file.stem}_elev{elev}_azim{azim}.png"
        image_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_file, image)

        mask_file = masks_path / f"{obj_file.stem}_elev{elev}_azim{azim}.png"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(mask_file, mask)
