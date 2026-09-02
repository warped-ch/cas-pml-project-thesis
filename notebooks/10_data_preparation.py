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
from datetime import datetime
from pathlib import Path

import cv2
import torch
import yaml
from joblib import Parallel, delayed
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

# %% [markdown]
# ## Dataset: Teeth3DS+
#
# - Download [Teeth3DS](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Teeth3DS.html) dataset from PyTorch Geometric.
# - Clean up leftovers from `3DTeethLand_challenge` split.

# %%
dataset_path_3d = root_path / config.get("dataset_path_3d")
dataset_path_3d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_3d={dataset_path_3d}")

# check if dataset_path_3d is empty to trigger initial Teeth3DS setup
if dataset_path_3d.is_dir() and not any(dataset_path_3d.iterdir()):
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
else:
    print(
        f"ℹ️ dataset_path_3d exists, skipping download (dataset_path_3d={dataset_path_3d})"
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
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

projections_path = root_path / f"{config['data_path_projections']}_{timestamp}"
projections_path.mkdir(parents=True, exist_ok=True)
print(f"projections_path={projections_path}")

images_path = projections_path / "images"
images_path.mkdir(parents=True, exist_ok=True)
print(f"images_path={images_path}")

masks_path = projections_path / "masks"
masks_path.mkdir(parents=True, exist_ok=True)
print(f"masks_path={masks_path}")

class_id_map = config["class_id_map"]

view_proj = view_projector.ViewProjector(config, device)

obj_files = list(dataset_path_3d.rglob("*.obj"))


def process_mesh(obj_file):
    mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)
    vertex_labels = file_io.load_vertex_labels(
        obj_file.with_suffix(".json"), class_id_map
    )

    images, masks = view_proj.render_2d_views(mesh, vertex_labels)

    for image, mask, view in zip(images, masks, view_proj.views):
        elev, azim = view[0], view[1]

        file_prefix = f"{obj_file.stem}_elev{elev}_azim{azim}.png"

        image_file = images_path / file_prefix
        image_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_file), image)

        mask_file = masks_path / file_prefix
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_file), mask)


# tuned for max throughput with 16 GB VRAM, safety first
Parallel(n_jobs=3)(
    delayed(process_mesh)(obj_file)
    for obj_file in tqdm(obj_files, desc="Rendering 2D views")
)
