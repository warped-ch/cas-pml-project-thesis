import numpy as np
import trimesh


class PostProcessing:
    def __init__(self, mesh_verts: np.ndarray, mesh_faces: np.ndarray):
        self.mesh = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)

    def keep_largest_components(
        self, vertex_labels: np.ndarray, background_label: int = 0
    ):
        """
        For each tooth label, keep only the largest connected cluster of vertices.
        """
        cleaned_labels = np.full_like(vertex_labels, background_label)
        unique_teeth = np.unique(vertex_labels)
        # ignore background
        unique_teeth = unique_teeth[unique_teeth != background_label]

        for tooth_id in unique_teeth:
            # Get all tooth vertices
            tooth_indices = np.where(vertex_labels == tooth_id)[0]
            if len(tooth_indices) == 0:
                continue

            # Filter mesh edges:
            # keep only edges where both vertices have this tooth_id
            edges = self.mesh.edges_unique
            mask = vertex_labels == tooth_id
            edge_mask = mask[edges[:, 0]] & mask[edges[:, 1]]
            tooth_edges = edges[edge_mask]

            # Find connected components:
            # pass the nodes list to ensure isolated vertices are counted as components
            components = trimesh.graph.connected_components(
                edges=tooth_edges, nodes=tooth_indices
            )

            if len(components) > 0:
                # find the largest component (with the most vertices)
                largest_comp = max(components, key=len)
                # keep only the largest component
                cleaned_labels[largest_comp] = tooth_id

        return cleaned_labels
