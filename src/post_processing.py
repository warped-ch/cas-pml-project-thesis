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

    def fill_holes(self, vertex_labels: np.ndarray, background_label: int = 0):
        """
        Fill holes (background segments that are fully enclosed by a tooth).
        """
        filled_labels = vertex_labels.copy()

        unique_teeth = np.unique(vertex_labels)
        # ignore background
        unique_teeth = unique_teeth[unique_teeth != background_label]

        edges = self.mesh.edges_unique

        for tooth_id in unique_teeth:
            # get all vertices which don't belong to the current tooth
            other_indices = np.where(vertex_labels != tooth_id)[0]

            # find connected components excluding the current tooth
            other_mask = vertex_labels != tooth_id
            other_edges = edges[other_mask[edges[:, 0]] & other_mask[edges[:, 1]]]
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
                # only fill true holes (background_label)
                component_list = list(component)
                if np.all(vertex_labels[component_list] == background_label):
                    filled_labels[component_list] = tooth_id

        return filled_labels
