# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: project-thesis (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %%
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d as o3d
import pandas as pd
import seaborn as sns
import yaml
from tqdm.auto import tqdm

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# %%
# FDI two-digit notation

# upper (mandibular), right (11-18), left (21-28)
fdi_upper_right_to_left = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
# lower (maxillary), right (41-48), left (31-38)
fdi_lower_right_to_left = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


# %%
def get_missing_teeth(vertex_labels, vertex_label_file):
    if vertex_labels is None:
        return None

    # check lower jaw
    if np.any(np.isin(fdi_lower_right_to_left, vertex_labels)):
        fdi_set = set(fdi_lower_right_to_left)
    # check upper jaw
    elif np.any(np.isin(fdi_upper_right_to_left, vertex_labels)):
        fdi_set = set(fdi_upper_right_to_left)
    else:
        print(f"⚠️ get_missing_teeth: invalid jaw (vertex_label_file={vertex_label_file})")
        return None

    missing_teeth = set(fdi_set).difference(vertex_labels)
    return sorted(missing_teeth)


# %%
def has_model_base(obj_file):
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
    #print(f"has_model_base: planarity={planarity}")

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
            window_name=f"Base Detection {str(obj_file.name)})"
        )

    # If planarity is very low, it's most probably a flat model base
    return planarity < 1e-6

# print(f"has_model_base: {has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\lower\DNSRP767\DNSRP767_lower.obj"))}")
# print(f"has_model_base: {has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\upper\DNSRP767\DNSRP767_upper.obj"))}")
# print(f"has_model_base: {has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\lower\018XZVD6\018XZVD6_lower.obj"))}")
# print(f"has_model_base: {has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\upper\018XZVD6\018XZVD6_upper.obj"))}")


# %%
def load_sample(obj_file):
    vertex_label_file = obj_file.with_suffix(".json")
    if not vertex_label_file.exists():
        print(f"⚠️ vertex_label_file does not exist: '{vertex_label_file}'")
        return None

    mesh_has_material = False
    with open(obj_file, 'r', encoding='utf-8') as file:
        content = file.read()
        if "mtl" in content.lower():
            mesh_has_material= True
            print(f"Mesh has material: {obj_file}")

    json_data = {}
    if vertex_label_file:
        with open(vertex_label_file, "r") as json_file:
            json_data = json.load(json_file)

    vertex_labels = np.array(json_data["labels"]) if "labels" in json_data else None

    return {
        "obj_file": str(obj_file),
        "id_patient": json_data.get("id_patient", None),
        "jaw": "lower" if "lower" in obj_file.name else "upper",
        "num_vertex_labels": len(vertex_labels) if vertex_labels is not None else pd.NA,
        "num_gingiva_vertex_labels": np.sum(vertex_labels == 0) if vertex_labels is not None else pd.NA,
        "num_tooth_vertex_labels": np.sum(vertex_labels != 0) if vertex_labels is not None else pd.NA,
        "missing_teeth": get_missing_teeth(vertex_labels, vertex_label_file),
        "has_model_base": has_model_base(obj_file),
        "mesh_has_material": mesh_has_material,
    }


# %%
# load the data

dataset_path = root_path / config.get("dataset_path_3d")
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

obj_files = list(dataset_path.rglob("*.obj"))

with ThreadPoolExecutor() as executor:
    data = list(
        tqdm(
            executor.map(load_sample, obj_files),
            total=len(obj_files),
            desc="Creating DataFrame",
        )
    )

# Remove None values from the list
data = [d for d in data if d is not None]

# %%
# create the DataFrame

df = pd.DataFrame(data)

# automatic type converion (automatically converts None/NaN into pd.NA for integer columns)
df = df.convert_dtypes()
df.to_csv(str(dataset_path.parent / "eda.csv"), index=False)

df.info()
df.head()

# %%
# save required metadata

meta_file = root_path / config.get("metadata")
df_meta = df[["obj_file", "id_patient", "jaw", "has_model_base"]]
df_meta.to_csv(str(meta_file), index=False)

# %% [markdown]
# ## Vertex label distribution

# %%
sns.histplot(df, x="num_vertex_labels", hue="jaw")
plt.show()

df.groupby("jaw")["num_vertex_labels"].describe()

# %%
sns.histplot(df, x="num_gingiva_vertex_labels", hue="jaw")
plt.show()

df.groupby("jaw")["num_gingiva_vertex_labels"].describe()

# %%
sns.histplot(df, x="num_tooth_vertex_labels", hue="jaw")
plt.show()

df.groupby("jaw")["num_tooth_vertex_labels"].describe()

# %%
# filter out rows where 'missing_teeth' is NaN/None
df_missing_teeth = df[df["missing_teeth"].notna()].copy()
# explode the list so each missing tooth gets its own row
df_missing_teeth = df_missing_teeth.explode("missing_teeth")
# reset the index to remove duplicate indices caused by exploding
df_missing_teeth = df_missing_teeth.reset_index(drop=True)

plt.figure(figsize=(12, 6))
sns.histplot(df_missing_teeth, x="missing_teeth", hue="jaw", binwidth=1, discrete=True)
plt.xticks(range(11, 49), rotation=45)
plt.show()

df.groupby("jaw")["missing_teeth"].describe()

# %% [markdown]
# ## Model / Mesh properties

# %%
# balace of models with and without model base

sns.countplot(df, x="has_model_base", hue="jaw")
plt.show()

df.groupby("jaw")["has_model_base"].describe()

# check consistency
has_base_mismatches = df.groupby('id_patient').filter(lambda x: x['has_model_base'].nunique() > 1)
if len(has_base_mismatches):
    has_base_mismatches.info()
    has_base_mismatches.head()

# %%
# check if obj mesh already has color/material attributes

df["mesh_has_material"].describe()

# %% [markdown]
# ## Potential dataset inconsistencies

# %%
# only lower or upper jaw in dataset

# get the set of unique patient IDs for each jaw type
ids_lower = set(df[df['jaw'] == 'lower']['id_patient'])
ids_upper = set(df[df['jaw'] == 'upper']['id_patient'])
print(f"ids_lower={len(ids_lower)}, ids_upper={len(ids_upper)}")

# identify the "missing" cases (only upper or lower jaw present in the dataset)
missing_ids_upper = list(ids_lower - ids_upper)
missing_ids_lower = list(ids_upper - ids_lower)
print(f"missing_ids_upper={missing_ids_upper}")
print(f"missing_ids_lower={missing_ids_lower}")
