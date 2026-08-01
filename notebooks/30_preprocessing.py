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
import math
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

pv.set_jupyter_backend('trame')
import torch
import yaml

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


# %%
def plot_mesh(mesh, vertex_labels=None):
    # Extract vertices and faces to CPU NumPy arrays
    # PyTorch3D stores faces as a tensor of shape (F, 3)
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()

    # Format faces for PyVista
    # PyVista requires a flat array where each polygon is prefixed by its number of padding vertices: [num_verts, v1, v2, v3, ...]
    num_faces = faces.shape[0]
    padding = torch.full((num_faces, 1), 3).numpy()  # Column of 3s since they are triangles
    faces_pv = torch.hstack([torch.tensor(padding), torch.tensor(faces)]).ravel().numpy()

    # Create the PyVista PolyData object
    pv_mesh = pv.PolyData(verts, faces_pv)

    if vertex_labels is not None:
        pv_mesh.point_data["labels"] = vertex_labels
        pv_mesh.color_labels(
            colors="viridis",
            scalars="labels",
            inplace=True
        )

    # Render the mesh inside the notebook
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv_mesh,
        color="lightgrey",
        scalars="labels" if vertex_labels is not None else None,
        show_scalar_bar=False,
        smooth_shading=True
    )
    plotter.show()


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

plot_mesh(mesh, vertex_labels)

# %%
# render 2D projection

preprocessing = preprocessing.Preprocessing(config, device)

images = preprocessing.render_2d_images(mesh)

# each row in views is [elevation, azimuth]
views = np.array(config["2d_projection"]["views"])

temp_out_path = dataset_path.parent / "temp" / obj_file.stem
print(f"temp_out_path={temp_out_path}")
temp_out_path.mkdir(parents=True, exist_ok=True)

cols = min(3, views.shape[0])
rows = math.ceil(views.shape[0] / cols)
f, axarr = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axarr.flat):
    if i < images.shape[0]:
        image = images[i]

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(image, cmap="gray")
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        cv2.imwrite(temp_out_path / f"view_elev{elev}_azim{azim}.png", image)

plt.tight_layout()
plt.show()

# %%
# render 2D projection label masks

segmentation_masks = preprocessing.render_2d_masks(mesh, vertex_labels)
print(f"segmentation_masks.shape={segmentation_masks.shape}")

# visualize segmentation mask

cmap = plt.colormaps['viridis'].copy().with_extremes(bad="white")

cols = min(3, views.shape[0])
rows = math.ceil(views.shape[0] / cols)
f, axarr = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axarr.flat):
    if i < images.shape[0]:
        segmentation_mask = segmentation_masks[i].astype(float)
        segmentation_mask[segmentation_mask == config["2d_projection"]["background_value"]] = np.nan  # hide background (remains white/transparent)

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(segmentation_mask, cmap=cmap, interpolation="nearest")
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        cv2.imwrite(temp_out_path / f"mask_elev{elev}_azim{azim}.png", segmentation_masks[i])

plt.tight_layout()
plt.show()
