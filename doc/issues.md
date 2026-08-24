## Issues and Workarounds

### Technical Issues

#### Backface Culling

see sample images: [backface_culling](/doc/backface_cullling/)

Status with current git commit: 578b3e60

| backface_culling | view image | mask image | color mask image |
|:---:|---|:---:|---|
| false | inside of the mesh is just flat grey (no features) | ok | ok |
| true | inside of mesh not visible (parts of the object just missing) | no label on the inside of the mesh (parts of the object just missing/white) | no label on the inside of the mesh (parts of the object just missing/white) |

examples:

- SL5I9AXM_lower.obj

### Fiftyone

#### All the labels have the same color

In case all the labels from different class IDs are displayed with the same color, got to `Color settings` in fiftyone app (the color palette icon) and set `Color annotations by` to `label`. 

### Open3D

#### 🛑 Blocker: Open3D web_visualizer crashes Jupyter kernel

Open3D (currently 0.19.0) requires Python 3.12 (currently latest 3.12 is 3.12.13). 
However, using web_visualizer backend for Jupyter crashes the Jupyter kernel:

```py
import open3d as o3d

mesh = o3d.io.read_triangle_mesh(obj_file)
o3d.web_visualizer.draw([mesh], width=800, height=600)
```

#### 🛑 Blocker: Open3D OffscreenRenderer: EGL Headless is not supported on this platform.

- https://github.com/isl-org/Open3D/issues/5307

> RuntimeError: [Open3D Error] (__cdecl open3d::visualization::rendering::EngineInstance::EngineInstance(void)) D:\a\Open3D\Open3D\cpp\open3d\visualization\rendering\filament\FilamentEngine.cpp:104: EGL Headless is not supported on this platform.

```py
import open3d as o3d

renderer = o3d.visualization.rendering.OffscreenRenderer(512, 512)
```

### PyTorch3D

#### miropsota - Torch Packages Compiler Repository

- https://github.com/MiroPsota/torch_packages_builder

### uv

#### Python versions

- https://docs.astral.sh/uv/concepts/python-versions/

- list installed python versions
    ```
    uv python list
    ```

- uninstall specific python version
    ```
    uv python uninstall cpython-3.12.13-windows-x86_64-none
    ```
