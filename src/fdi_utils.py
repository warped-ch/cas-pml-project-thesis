import numpy as np

# Upper Jaw (Maxillary)
# Sequence: Right (18-11) to Left (21-28)
FDI_UPPER_RIGHT_TO_LEFT = [
    18, 17, 16, 15, 14, 13, 12, 11,  # Quadrant 1
    21, 22, 23, 24, 25, 26, 27, 28   # Quadrant 2
]

# Lower Jaw (Mandibular)
# Sequence: Right (48-41) to Left (31-38)
FDI_LOWER_RIGHT_TO_LEFT = [
    48, 47, 46, 45, 44, 43, 42, 41,  # Quadrant 4
    31, 32, 33, 34, 35, 36, 37, 38   # Quadrant 3
]

FDI_UPPER_SET = set(FDI_UPPER_RIGHT_TO_LEFT)
FDI_LOWER_SET = set(FDI_LOWER_RIGHT_TO_LEFT)
FDI_ALL_TEETH = sorted(FDI_UPPER_SET | FDI_LOWER_SET)

def get_missing_teeth(vertex_labels, vertex_label_file: str) -> set[int]:
    if vertex_labels is None:
        print(f"⚠️ get_missing_teeth: no vertex labels (vertex_label_file={vertex_label_file})")
        return None

    # check lower jaw
    if np.any(np.isin(FDI_LOWER_RIGHT_TO_LEFT, vertex_labels)):
        fdi_set = FDI_LOWER_SET
    # check upper jaw
    elif np.any(np.isin(FDI_UPPER_RIGHT_TO_LEFT, vertex_labels)):
        fdi_set = FDI_UPPER_SET
    else:
        print(f"⚠️ get_missing_teeth: invalid jaw (vertex_label_file={vertex_label_file})")
        return None

    missing_teeth = fdi_set.difference(vertex_labels)
    return sorted(missing_teeth)
