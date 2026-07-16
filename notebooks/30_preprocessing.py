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
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import yaml
from pathlib import Path
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    look_at_view_transform,
    camera_position_from_spherical_angles,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
)

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# %%
import pyvista as pv

pv.set_jupyter_backend('trame')

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

dataset_path = root_path / config["dataset_path"]
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

# TODO: use train/test/val splits instead
obj_files = list(dataset_path.rglob("*.obj"))

obj_file = random.choice(obj_files)
print(f"obj_file={obj_file}")

mesh = IO().load_mesh(obj_file, device=device)

# align mesh to origin
mesh_center = mesh.verts_packed().mean(dim=0)
mesh = mesh.offset_verts(-mesh_center)

# load the vertex labels
json_file = obj_file.with_suffix(".json")
print(f"json_file={json_file}")
with open(json_file, "r") as json_file:
    json_data = json.load(json_file)
vertex_labels = np.array(json_data["labels"], dtype=np.uint8)

plot_mesh(mesh, vertex_labels)

# %%
# render 2D projection

# each row in views is [elevation, azimuth]
views = np.array(config["2d_projection"]["views"])

R, T = look_at_view_transform(
    dist=config["2d_projection"]["distance"],
    elev=views[:, 0],
    azim=views[:, 1],
    device=device
)

cameras = FoVPerspectiveCameras(
    znear=0.1,
    zfar=100.0,
    fov=config["2d_projection"]["fov"],
    R=R,
    T=T,
    device=device
)

raster_settings = RasterizationSettings(
    image_size=config["2d_projection"]["image_size"]
)

light_dir = camera_position_from_spherical_angles(distance=1.0, elevation=views[:, 0], azimuth=views[:, 1], device=device)

lights = DirectionalLights(
    direction=light_dir,
    ambient_color=((0.5, 0.5, 0.5),),  
    diffuse_color=((0.5, 0.5, 0.5),),  
    specular_color=((0.2, 0.2, 0.2),), # reduced specular highlight to keep tooth boundaries matte
    device=device
)

blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        cameras=cameras,
        lights=lights,
        blend_params=blend_params,
        device=device
    )
)

# define a default color for each vertex
num_vertices = mesh.verts_packed().shape[0]
verts_features = torch.ones((1, num_vertices, 3), dtype=torch.float32, device=device) * 0.75
mesh.textures = TexturesVertex(verts_features=verts_features)

meshes = mesh.extend(views.shape[0])
images = renderer(meshes)

temp_out_path = dataset_path.parent / "temp" / obj_file.stem
print(f"temp_out_path={temp_out_path}")
temp_out_path.mkdir(parents=True, exist_ok=True)

cols = min(3, views.shape[0])
rows = math.ceil(views.shape[0] / cols)
f, axarr = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axarr.flat):
    if i < images.shape[0]:
        # slice [..., :3] to remove alpha channel
        img_np = images[i].detach().cpu().numpy()[..., :3]

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(img_np)
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        plt.imsave(temp_out_path / f"view_2d_elev{elev}_azim{azim}.png", img_np)

plt.tight_layout()
plt.show()

# %%
# render 2D projection label masks (face index rasterization)

# Instead of rendering colors and guessing pixels, this method uses PyTorch3D’s rasterizer to determine exactly 
# which face index is visible at every pixel. It then maps that face back to its vertex labels.

mask_raster_settings = RasterizationSettings(
    image_size=config["2d_projection"]["image_size"],
    blur_radius=0.0,
    faces_per_pixel=1,
)
mask_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=mask_raster_settings)

# get fragments (pix_to_face contains the face index for each pixel)
fragments = mask_rasterizer(meshes)
# Shape: (N, H, W) -> values are face indices, -1 means background
pix_to_face = fragments.pix_to_face[..., 0]

# map faces to vertex labels (identical for all view since it's the same mesh)
# faces shape: (F, 3) | vertex_labels_tensor shape: (V,)
faces = mesh.faces_packed()
vertex_labels_tensor = torch.from_numpy(vertex_labels).to(mesh.device)

# get the labels of the 3 vertices for every face -> Shape: (F, 3)
face_vert_labels = vertex_labels_tensor[faces]

# TODO, which method?
# define face label by taking the first vertex
face_labels = face_vert_labels[:, 0] # Shape: (F,)
# define face label by majority vote (or taking the first vertex)
#face_labels = torch.mode(face_vert_labels, dim=1).values # Shape: (F,)
face_labels = face_labels.repeat(views.shape[0])

# add a background label
background_label = 255
face_labels_with_bg = torch.cat([face_labels, torch.tensor([background_label], device=mesh.device)])

# generate the final 2D segmentation mask
segmentation_masks = face_labels_with_bg[pix_to_face].cpu().numpy().astype(np.uint8)

# visualize segmentation mask

cmap = plt.colormaps['viridis'].with_extremes(bad="white")

cols = min(3, views.shape[0])
rows = math.ceil(views.shape[0] / cols)
f, axarr = plt.subplots(rows, cols, figsize=(12, 12))
for i, ax in enumerate(axarr.flat):
    if i < images.shape[0]:
        segmentation_mask = segmentation_masks[i].astype(float)
        segmentation_mask[segmentation_mask == background_label] = np.nan  # hide background (remains white/transparent)

        elev = views[i, 0].item()
        azim = views[i, 1].item()

        ax.imshow(segmentation_mask, cmap=cmap, interpolation="nearest")
        ax.set_title(f"elevation={elev}, azimuth={azim}")

        #plt.imsave(temp_out_path / f"view_2d_elev{elev}_azim{azim}.png", segmentation_masks[i])
        img = Image.fromarray(segmentation_masks[i])
        img.save(temp_out_path / f"view_2d_mask_elev{elev}_azim{azim}.png")

plt.tight_layout()
plt.show()
