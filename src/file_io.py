import json
import numpy as np


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
