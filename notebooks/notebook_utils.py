import math

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from matplotlib.colors import Colormap
from pytorch3d.structures import Meshes

pv.set_jupyter_backend("trame")


def plot_image_grid(
    images: np.ndarray,
    background_label: float | None = None,
    titles: list[str] | None = None,
    columns: int = 3,
    cmap: str | Colormap | None = None,
) -> None:
    num_images = images.shape[0]
    cols = min(columns, num_images)
    rows = math.ceil(num_images / cols)
    fig, axarr = plt.subplots(rows, cols, figsize=(12, 12))
    for i, ax in enumerate(axarr.flat):
        if i < num_images:
            image = images[i]
            if background_label is not None:
                image = image.astype(float)
                image[image == background_label] = np.nan

            ax.imshow(image, cmap=cmap)
            if titles:
                ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def plot_mesh(mesh: Meshes, vertex_labels: np.ndarray = None):
    # Extract vertices and faces to CPU NumPy arrays
    # PyTorch3D stores faces as a tensor of shape (F, 3)
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()

    # Format faces for PyVista
    # PyVista requires a flat array where each polygon is prefixed by its number of padding vertices: [num_verts, v1, v2, v3, ...]
    num_faces = faces.shape[0]
    padding = torch.full(
        (num_faces, 1), 3
    ).numpy()  # Column of 3s since they are triangles
    faces_pv = (
        torch.hstack([torch.tensor(padding), torch.tensor(faces)]).ravel().numpy()
    )

    # Create the PyVista PolyData object
    pv_mesh = pv.PolyData(verts, faces_pv)

    if vertex_labels is not None:
        pv_mesh.point_data["labels"] = vertex_labels
        pv_mesh.color_labels(colors="viridis", scalars="labels", inplace=True)

    # Render the mesh inside the notebook
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv_mesh,
        color="lightgrey",
        scalars="labels" if vertex_labels is not None else None,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    plotter.show()
