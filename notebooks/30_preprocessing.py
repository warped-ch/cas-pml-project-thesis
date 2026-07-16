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
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import yaml
from pathlib import Path
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

def plot_mesh(mesh):
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

    # Render the mesh inside the notebook
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color="lightblue")
    plotter.show()


# %%
# load a mesh sample

dataset_path = root_path / config["dataset_path"]
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

obj_files = list(dataset_path.rglob("*.obj"))

obj_file = random.choice(obj_files)
print(f"obj_file={obj_file}")

mesh = IO().load_mesh(obj_file, device=device)
if mesh.textures is None:
    num_vertices = mesh.verts_packed().shape[0]
    # define a default color for each vertex: [1, num_vertices, 3] -> batch size 1
    verts_features = torch.ones((1, num_vertices, 3), dtype=torch.float32, device=device) * 0.75
    mesh.textures = TexturesVertex(verts_features=verts_features)

# align mesh to origin
mesh_center = mesh.verts_packed().mean(dim=0)
mesh = mesh.offset_verts(-mesh_center)

plot_mesh(mesh)

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

        plt.imsave(temp_out_path / f"view_2d_evev{elev}_azim{azim}.png", img_np)

plt.tight_layout()
plt.show()
