import math
from typing import Any

import glasbey
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import supervision as sv
from pytorch3d.structures import Meshes

pv.set_jupyter_backend("trame")


def get_colors(palette: Any = "tab10", palette_size: int = 17) -> list[str]:
    """
    Returns a list of color strings (hex, RGB).

    Args:
        palette_size: matches the number of class IDs per jaw (lower/upper) by default.
    """
    colors = glasbey.extend_palette(palette, palette_size)

    # find the "greyest" color and move it to the front (gingiva, class_0)
    rgb_arr = mcolors.to_rgba_array(colors)[:, :3]
    hsv_arr = mcolors.rgb_to_hsv(rgb_arr)
    greyest_idx = np.argmin(hsv_arr[:, 1])  # find index of lowest saturation
    colors.insert(0, colors.pop(greyest_idx))

    return colors


def convert_colors_pv(colors: list[str], class_ids: list[int]) -> dict[int, str]:
    """
    Maps class IDs to colors based on their index position.

    Args:
        colors: A list of color strings (hex, RGB).
        class_ids: A list of unique class IDs (sparse FDI labels).

    Returns:
        A color dict mapping each class ID (int) to a color string (hex, RGB).
    """
    color_dict = {}
    for class_id in class_ids:
        # Find the class_idx that matches Supervision's class_id (0-based, continuous)
        class_idx = class_ids.index(int(class_id))
        color_idx = class_idx % len(colors)
        color_dict[int(class_id)] = str(colors[color_idx])
    return color_dict


def convert_colors_sv(colors: list[str]) -> sv.ColorPalette:
    def hex_rgb_to_hex_bgr(hex_str):
        # Standard Hex: #RRGGBB -> #BBGGRR
        return f"#{hex_str[5:7]}{hex_str[3:5]}{hex_str[1:3]}"

    # TODO: bug in Supervision?
    # https://supervision.roboflow.com/draw/color/#colorpalette
    colors_bgr = [hex_rgb_to_hex_bgr(c) for c in colors]
    return sv.ColorPalette.from_hex(colors_bgr)


def plot_histogram_grid(
    images: np.ndarray | list,
    background_value: int | None = None,
    bins: int = np.iinfo(np.uint8).max + 1,
    titles: list[str] | None = None,
    columns: int = 3,
) -> None:
    if isinstance(images, list):
        images = np.array(images)

    num_images = images.shape[0]
    cols = min(columns, num_images)
    rows = math.ceil(num_images / cols)

    fig, axarr = plt.subplots(rows, cols, figsize=(12, 12))
    for i, ax in enumerate(axarr.flat):
        if i < num_images:
            values = images[i].flatten()
            if background_value is not None:
                values = values[values != background_value]
            ax.hist(values, bins=bins, range=(0, bins))
            ax.set_xlim(0, bins)
            if titles:
                ax.set_title(titles[i])
        else:
            # hide unused plots
            ax.axis("off")
            ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_image_grid(
    images: np.ndarray | list,
    background_label: float | None = None,
    titles: list[str] | None = None,
    columns: int = 3,
    cmap: str | mcolors.Colormap | None = None,
) -> None:
    if isinstance(images, list):
        images = np.array(images)

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
        else:
            # hide unused plots
            ax.axis("off")
            ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_mesh(
    mesh: Meshes, vertex_labels: np.ndarray = None, colors: dict[int, str] | None = None
):
    # Extract vertices and faces to CPU NumPy arrays
    # PyTorch3D stores faces as a tensor of shape (F, 3)
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()

    # Format faces for PyVista [3, v1, v2, v3, ...]
    num_faces = faces.shape[0]
    padding = np.full((num_faces, 1), 3)
    faces_pv = np.hstack([padding, faces]).ravel()

    # Create the PyVista PolyData object
    pv_mesh = pv.PolyData(verts, faces_pv)

    if vertex_labels is not None:
        pv_mesh.point_data["labels"] = vertex_labels
        pv_mesh.color_labels(colors=colors, scalars="labels", inplace=True)

    # Render the mesh inside the notebook
    plotter = pv.Plotter()
    plotter.add_mesh(
        pv_mesh,
        color="lightgrey",
        scalars="labels_rgb" if vertex_labels is not None else None,
        rgb=True if vertex_labels is not None else False,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    plotter.show()
