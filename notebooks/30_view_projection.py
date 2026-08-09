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

# %%
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import notebook_utils as nb_utils
import numpy as np
import torch
import yaml

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

# %%
# load a mesh sample

dataset_path = root_path / config["dataset_path_3d"]
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

# TODO: use train/test/val splits instead
obj_files = list(dataset_path.rglob("*.obj"))

obj_file = random.choice(obj_files)
print(f"obj_file={obj_file}")
mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)

# load the vertex labels
json_file = obj_file.with_suffix(".json")
print(f"json_file={json_file}")
vertex_labels = file_io.load_vertex_labels(json_file, config["class_id_map"])

nb_utils.plot_mesh(mesh, vertex_labels)

# %%
# render 2D projection

view_proj = view_projector.ViewProjector(config, device)

images = view_proj.render_2d_images_np(mesh)

# each row in views is [elevation, azimuth]
views = np.array(config["2d_projection"]["views"])

nb_utils.plot_image_grid(
    images=images,
    background_label=None,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
    cmap="gray"
)

temp_out_path = dataset_path.parent / "temp" / obj_file.stem
print(f"temp_out_path={temp_out_path}")
temp_out_path.mkdir(parents=True, exist_ok=True)

for i, view in enumerate(views):
    cv2.imwrite(temp_out_path / f"view_elev{view[0]}_azim{view[1]}.png", images[i])

# %%
# render 2D projection label masks

segmentation_masks = view_proj.render_2d_masks(mesh, vertex_labels)
print(f"segmentation_masks.shape={segmentation_masks.shape}")

# visualize segmentation masks

nb_utils.plot_image_grid(
    images=segmentation_masks,
    background_label=config["2d_projection"]["background_value"],
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
    cmap=plt.colormaps['viridis'].copy().with_extremes(bad="white")
)

nb_utils.plot_image_grid(
    images=segmentation_masks,
    background_label=None,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
    cmap="grey",
)

for i, view in enumerate(views):
    cv2.imwrite(temp_out_path / f"mask_elev{view[0]}_azim{view[1]}.png", segmentation_masks[i])

# %%
# roundtrip: back projection of ground truth masks to mesh vertex labels

vertex_labels_out = view_proj.back_project_vertex_labels(mesh, segmentation_masks)
print(f"vertex_labels.shape={vertex_labels.shape}")
print(f"vertex_labels_out.shape={vertex_labels_out.shape}")

nb_utils.plot_mesh(mesh, vertex_labels_out)
