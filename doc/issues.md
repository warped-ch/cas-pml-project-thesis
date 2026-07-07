## Issues and Workarounds

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
