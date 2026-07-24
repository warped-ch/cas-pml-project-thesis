import json

import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.structures import Meshes


def load_mesh_origin_aligned(
    obj_file: str, device: str | torch.device | None = None
) -> Meshes:
    """Loads a 3D mesh from an OBJ file and centers it at the coordinate origin.

    Args:
        obj_file: The file path to the 3D mesh file (.obj).
        device: The target device for computation (e.g., 'cuda', 'cpu', or
          None).

    Returns:
        Meshes: The loaded and origin-aligned PyTorch3D Mesh object.
    """
    mesh = IO().load_mesh(obj_file, device=device)

    # align mesh to origin
    mesh_center = mesh.verts_packed().mean(dim=0)
    mesh = mesh.offset_verts(-mesh_center)

    return mesh


def load_vertex_labels(json_file: str) -> np.ndarray:
    """
    Loads vertex labels from a JSON file and converts them into a NumPy array of unsigned 8-bit integers.

    Args:
        json_file (str): The path to the JSON file containing the label data.

    Returns:
        np.ndarray: A NumPy array containing the vertex labels.
    """
    with open(json_file, "r") as f:
        json_data = json.load(f)

    return np.array(json_data["labels"], dtype=np.uint8)
