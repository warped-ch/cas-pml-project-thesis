from typing import Any

import numpy as np
import trimesh


class PostProcessing:
    def __init__(
        self, config: dict[str, Any], mesh_verts: np.ndarray, mesh_faces: np.ndarray
    ):
        self.mesh = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
        self.edges_unique = self.mesh.edges_unique

        self.background_value = config["2d_projection"]["background_value"]

    def keep_largest_components(self, vertex_labels: np.ndarray):
        """
        For each tooth label, keep only the largest connected cluster of vertices.
        """
        cleaned_labels = np.full_like(vertex_labels, self.background_value)

        unique_teeth = np.unique(vertex_labels)
        # ignore background
        unique_teeth = unique_teeth[unique_teeth != self.background_value]

        for tooth_id in unique_teeth:
            # Get all tooth vertices
            tooth_indices = np.where(vertex_labels == tooth_id)[0]
            if len(tooth_indices) == 0:
                continue

            # Filter mesh edges:
            # keep only edges where both vertices have this tooth_id
            mask = vertex_labels == tooth_id
            edge_mask = mask[self.edges_unique[:, 0]] & mask[self.edges_unique[:, 1]]
            tooth_edges = self.edges_unique[edge_mask]

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

    def fill_holes(self, vertex_labels: np.ndarray):
        """
        Fill holes (background segments that are fully enclosed by a tooth).
        """
        filled_labels = vertex_labels.copy()

        unique_teeth = np.unique(vertex_labels)
        # ignore background
        unique_teeth = unique_teeth[unique_teeth != self.background_value]

        for tooth_id in unique_teeth:
            # get all vertices which don't belong to the current tooth
            other_indices = np.where(vertex_labels != tooth_id)[0]

            # find connected components excluding the current tooth
            other_mask = vertex_labels != tooth_id
            other_edges = self.edges_unique[
                other_mask[self.edges_unique[:, 0]]
                & other_mask[self.edges_unique[:, 1]]
            ]
            other_components = trimesh.graph.connected_components(
                edges=other_edges, nodes=other_indices
            )
            if len(other_components) <= 1:
                # no holes for this tooth
                continue

            # the largest component is rest of the mesh, the part outside the current tooth boundary
            outside_component_idx = np.argmax([len(c) for c in other_components])

            # fill holes inside the tooth
            for i, component in enumerate(other_components):
                if i == outside_component_idx:
                    continue
                # only fill true holes (background_value)
                component_list = list(component)
                if np.all(vertex_labels[component_list] == self.background_value):
                    filled_labels[component_list] = tooth_id

        return filled_labels
