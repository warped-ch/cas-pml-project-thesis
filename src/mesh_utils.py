from pathlib import Path

import networkx as nx
import numpy as np
import open3d as o3d


def has_model_base(obj_file: Path) -> bool:
    if not obj_file.exists():
        print(f"⚠️ obj_file does not exist: {obj_file}")
    mesh = o3d.io.read_triangle_mesh(str(obj_file))

    # Get boundary edges (allow_boundary_edges=False returns boundary + non-manifold)
    boundary_edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    if len(boundary_edges) == 0:
        return True

    # Get the largest boundary chain
    G = nx.Graph()
    G.add_edges_from(boundary_edges)
    largest_indices = list(max(nx.connected_components(G), key=len))
    points = np.asarray(mesh.vertices)[largest_indices]

    # Convert boundary vertices into a PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # remove outliers  (e.g. for DNSRP767_upper.obj, 018XZVD6_upper.obj)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    points = points[ind]

    # PCA (using eigh for symmetric matrices, this avoids the "+0j" complex number issue)
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    cov = np.cov(centered_points.T)
    eigenvalues, _ = np.linalg.eigh(cov)

    # Planarity (smallest eigenvalue / sum of all)
    planarity = eigenvalues[0] / sum(eigenvalues)
    # print(f"has_model_base: planarity={planarity}")

    visualize = False
    if visualize:
        # Paint the main mesh grey and semi-transparent
        mesh.paint_uniform_color([0.8, 0.8, 0.8])
        # Paint Inliers green
        inlier_pcd = pcd.select_by_index(ind)
        inlier_pcd.paint_uniform_color([0, 1, 0])
        # Paint Outliers red
        outlier_pcd = pcd.select_by_index(ind, invert=True)
        outlier_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(
            [mesh, inlier_pcd, outlier_pcd],
            window_name=f"Base Detection {str(obj_file.name)})",
        )

    # If planarity is very low, it's most probably a flat model base
    return planarity < 1e-6
