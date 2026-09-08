# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
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
import seaborn as sns
import torch
import yaml
from matplotlib.colors import ListedColormap

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
colors = nb_utils.get_colors()
print(f"colors: {len(colors)}, {colors}")

class_ids = config["class_ids"]
sns.palplot(colors)
plt.title("FDI class_id color labels", fontsize=16, pad=20)
plt.xticks(range(len(colors)), class_ids)
plt.show()

colors_pv = nb_utils.convert_colors_pv(colors, config["class_ids"])
print(f"colors_pv: {len(colors_pv)}, {colors_pv}")

# %%
# load a mesh sample

dataset_path = root_path / config["dataset_path_3d"]
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

obj_files = list(dataset_path.rglob("*.obj"))

obj_file = random.choice(obj_files)
print(f"obj_file={obj_file}")
mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)

# load the vertex labels
json_file = obj_file.with_suffix(".json")
print(f"json_file={json_file}")
vertex_labels = file_io.load_vertex_labels(json_file)

nb_utils.plot_mesh(mesh, vertex_labels, colors_pv)

# %%
# render 2D projections (view images and masks)

view_proj = view_projector.ViewProjector(config, device)
images, masks = view_proj.render_2d_views(mesh, vertex_labels)

# each row in views is [elevation, azimuth]
views = np.array(config["2d_projection"]["views"])

# %% [markdown]
# ## Images

# %%
print(f"images.shape={images.shape}")

titles = [f"elevation={view[0]}, azimuth={view[1]}" for view in views]
nb_utils.plot_image_grid(
    images=images, background_label=None, titles=titles, cmap="gray"
)
nb_utils.plot_histogram_grid(
    images=images,
    background_value=config["2d_projection"]["background_value"],
    titles=titles,
)

temp_path_sample = root_path / config.get("temp_path") / obj_file.stem
print(f"temp_path_sample={temp_path_sample}")
temp_path_sample.mkdir(parents=True, exist_ok=True)

for i, view in enumerate(views):
    cv2.imwrite(temp_path_sample / f"view_elev{view[0]}_azim{view[1]}.png", images[i])

# %% [markdown]
# ## Masks

# %%
print(f"masks.shape={masks.shape}")

colors_cmap = ListedColormap(colors)
colors_cmap.set_bad(color="black")

# visualize segmentation masks
titles = [f"elevation={view[0]}, azimuth={view[1]}" for view in views]
nb_utils.plot_image_grid(
    images=masks,
    background_label=None,
    titles=titles,
    cmap="grey",
)
nb_utils.plot_image_grid(
    images=masks,
    background_label=config["2d_projection"]["background_value"],
    titles=titles,
    cmap=colors_cmap,
)

for i, view in enumerate(views):
    cv2.imwrite(temp_path_sample / f"mask_elev{view[0]}_azim{view[1]}.png", masks[i])

# %%
# roundtrip: back projection of ground truth masks to mesh vertex labels

vertex_labels_out = view_proj.back_project_vertex_labels(mesh, masks)
print(f"vertex_labels.shape={vertex_labels.shape}")
print(f"vertex_labels_out.shape={vertex_labels_out.shape}")

nb_utils.plot_mesh(mesh, vertex_labels_out, colors_pv)
