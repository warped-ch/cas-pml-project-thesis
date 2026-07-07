# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: project-thesis (3.12.13)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import open3d as o3d
import random
import yaml
from pathlib import Path

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# %%
def load_mesh(obj_file):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    return mesh


# %%
# load a mesh sample

dataset_path = root_path / config.get("dataset_path")
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

obj_files = list(dataset_path.rglob("*.obj"))

obj_file = random.choice(obj_files)
print(f"obj_file={obj_file}")

mesh = load_mesh(obj_file)

# align mesh to origin
mesh.translate(-mesh.get_center())

origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

# TODO: crashes Jupyter kernel...
# import open3d.web_visualizer as webvis
# webvis.draw([mesh, origin_axis], width=1280, height=1024)
o3d.visualization.draw_geometries(
    [mesh, origin_axis],
    window_name=f"{obj_file.name}",
    width=1280,
    height=1024,
    mesh_show_back_face=True,
)

# https://stackoverflow.com/questions/70273002/project-3d-mesh-on-2d-image-using-camera-intrinsic-matrix
img_width = 512
img_height = 512
# RuntimeError: [Open3D Error] (__cdecl open3d::visualization::rendering::EngineInstance::EngineInstance(void)) D:\a\Open3D\Open3D\cpp\open3d\visualization\rendering\filament\FilamentEngine.cpp:104: EGL Headless is not supported on this platform.
# https://github.com/isl-org/Open3D/issues/5307
renderer = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)
renderer.scene.set_background([0.0, 0.0, 0.0, 1.0]) # [r, g, b, a]

print(f"obj_file.name={obj_file.name}")
renderer.scene.add_geometry(obj_file.name, mesh)

aspect_ratio = img_width / img_height
fov = 60.0 # [deg]
fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
near_plane = 0.1
far_plane = 100.0
renderer.scene.camera.set_projection(fov, aspect_ratio, near_plane, far_plane, fov_type)

center_pos = [0, 0, 0]
eye_pos = [0, 10, 0]
up_pos = [0, 1, 0]
renderer.scene.camera.look_at(center_pos, eye_pos, up_pos)

img = renderer.render_to_image()
plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.show()
