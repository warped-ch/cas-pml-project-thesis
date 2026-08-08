import json
import random

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


def load_vertex_labels(
    json_file: str, class_id_map: dict[int, int] | None = None
) -> np.ndarray:
    """
    Loads vertex labels from a JSON file and converts them into a NumPy array of unsigned 8-bit integers.

    Args:
        json_file (str): The path to the JSON file containing the label data.
        class_id_map (dict, optional): A dictionary mapping old class IDs to new class IDs.
            Unmapped IDs will retain their original value.

    Returns:
        np.ndarray: A NumPy array containing the vertex labels.
    """
    with open(json_file, "r") as f:
        json_data = json.load(f)

    vertex_labels = np.array(json_data["labels"], dtype=np.uint8)

    if class_id_map:
        lookup = np.arange(256, dtype=np.uint8)
        for old_id, new_id in class_id_map.items():
            lookup[old_id] = new_id
        vertex_labels = lookup[vertex_labels]

    return vertex_labels

def read_random_line_from_file(file_path: str) -> str:
    """
    Reads a single random line from a file.

    https://stackoverflow.com/a/3540315
    
    Args:
        file_path (str): The path to the file.

    Returns:
        str: The content of one randomly selected line, stripped of whitespace.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        line = next(file)
        for num, aline in enumerate(file, 2):
            if random.randrange(num):
                continue
            line = aline
        return line.strip()
