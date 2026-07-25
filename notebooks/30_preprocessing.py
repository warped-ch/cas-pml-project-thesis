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
import supervision as sv
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
vertex_labels = file_io.load_vertex_labels(json_file)

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
        segmentation_mask = images[i]

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(segmentation_mask)
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        cv2.imwrite(temp_out_path / f"view_elev{elev}_azim{azim}.png", segmentation_mask)

plt.tight_layout()
plt.show()

# %%
# render 2D projection label masks

segmentation_masks = preprocessing.render_2d_masks(mesh, vertex_labels)
print(f"segmentation_masks.shape={segmentation_masks.shape}")

# visualize segmentation mask

background_value = config["2d_projection"]["background_value"]
cmap = plt.colormaps['viridis'].with_extremes(bad="white")

cols = min(3, views.shape[0])
rows = math.ceil(views.shape[0] / cols)
f, axarr = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axarr.flat):
    if i < images.shape[0]:
        segmentation_mask = segmentation_masks[i].astype(float)
        segmentation_mask[segmentation_mask == background_value] = np.nan  # hide background (remains white/transparent)

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(segmentation_mask, cmap=cmap, interpolation="nearest")
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        cv2.imwrite(temp_out_path / f"mask_elev{elev}_azim{azim}.png", segmentation_masks[i])

plt.tight_layout()
plt.show()

# %%
# Convert mask images to YOLO annotations

for i, segmentation_mask in enumerate(segmentation_masks):
    unique_values = np.unique(segmentation_mask)
    class_ids = [v for v in unique_values if v != background_value]

    mask_height, mask_width = segmentation_mask.shape[:2]
    scale_vector = np.array([mask_width, mask_height], dtype=np.float32)

    yolo_annotations = []

    # generate a boolean mask for each class_id
    for class_id in class_ids:
        binary_mask = (segmentation_mask == class_id)
        if sv.contains_holes(binary_mask):
            # TODO: handle holes (important especially for top down view, otherwise "tooth holes" will be filled)
            # https://github.com/roboflow/supervision/issues/574
            print(f"⚠️ binary_mask for class {class_id} contains holes!")

        polygons = sv.mask_to_polygons(binary_mask)
        for polygon in polygons:
            # normalize and clip all points
            normalized_poly = np.clip(polygon / scale_vector, 0.0, 1.0)
            # flatten the (N, 2) array into a 1D array: [x1, y1, x2, y2, ...]
            flattened_poly = normalized_poly.ravel()
            # string serialization
            coords_str = " ".join(f"{coord:.6f}" for coord in flattened_poly)
            yolo_annotations.append(f"{class_id} {coords_str}")

    elev = views[i, 0].item()
    azim = views[i, 1].item()
    yolo_txt = temp_out_path / f"mask_elev{elev}_azim{azim}.txt"
    with open(yolo_txt, "w") as f:
        f.write("\n".join(yolo_annotations))

# %%
# Roundtrip: load the yolo annotations and visualize on segmentation masks

annotated_images = []
for i, view in enumerate(views):
    segmentation_mask = segmentation_masks[i]

    elev = view[0]
    azim = view[1]
    yolo_txt = temp_out_path / f"mask_elev{elev}_azim{azim}.txt"
    #print(f"yolo_txt={yolo_txt}")

    class_ids = []
    polygons = []

    with open(yolo_txt, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        
        class_id = int(parts[0])
        poly_coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
        
        # scale coordinates back to pixel space
        img_height, img_width = segmentation_mask.shape[:2]
        scale_vector = np.array([img_width, img_height], dtype=np.float32)
        poly_coords = poly_coords * scale_vector

        class_ids.append(class_id)
        polygons.append(poly_coords.astype(np.int32))

    masks = [sv.polygon_to_mask(polygon, (img_width, img_height)) for polygon in polygons]
    stacked_masks = np.stack(masks, axis=0).astype(bool) # stack masks into a 3D boolean array of shape (N, H, W)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(stacked_masks),
        mask=stacked_masks,
        class_id=np.array(class_ids, dtype=np.int32)
    )

    ### debug start

    unique_classes = np.unique(detections.class_id)

    for cid in unique_classes:
        # Create a boolean mask of where the current class appears
        indices = (detections.class_id == cid)
        
        # Create a new Detections object containing only those instances
        class_detections = sv.Detections(
            xyxy=detections.xyxy[indices],
            mask=detections.mask[indices],
            class_id=detections.class_id[indices]
        )

        class_image = sv.MaskAnnotator().annotate(
            scene=cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2RGB), 
            detections=class_detections
        )

        print(f"cid={cid}")
        plt.imshow(class_image)
        plt.axis('off')
        plt.show()
    
    ### debug end

    # TODO: static color palette, otherwise colors might change based on which labels are present...
    # custom static viridis palette for fdi labels and gum
    color_count = len(set(class_ids))
    print(f"color_count={color_count}")
    color = sv.ColorPalette.from_matplotlib("viridis", color_count)

    print(f"class_ids={set(class_ids)}")
    print(f"color={color}")

    mask_annotator = sv.MaskAnnotator(
        color=color,
        # TODO: setting opacity too high somehow leads to one tooth being invisible and labeled as gum... ?
        # might be related to blob hole handling
        # opacity=1.0,
    )

    annotated_image = mask_annotator.annotate(
        scene=cv2.cvtColor(segmentation_mask, cv2.COLOR_GRAY2RGB), 
        detections=detections
    )
    annotated_images.append(annotated_image)

cols = min(3, views.shape[0])
rows = math.ceil(views.shape[0] / cols)
f, axarr = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axarr.flat):
    if i < len(annotated_images):
        annotated_image = annotated_images[i]

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(annotated_image)
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        plt.imsave(temp_out_path / f"mask_annotated_elev{elev}_azim{azim}.png", annotated_image)

plt.tight_layout()
plt.show()

# TODO: annotate view images as well

