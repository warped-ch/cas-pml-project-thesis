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
def load_sample(obj_file):
    vertex_label_file = obj_file.with_suffix(".json")
    if not vertex_label_file.exists():
        print(f"⚠️ vertex_label_file does not exist: '{vertex_label_file}'")
        return None

    # TODO: this won't work, some bases are open on the bottom...
    mesh_is_manifold = False
    # mesh = o3d.io.read_triangle_mesh(obj_file)
    # mesh_is_manifold = mesh.is_edge_manifold() and mesh.is_vertex_manifold()

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
        "id_patient": json_data.get("id_patient", None),
        "obj_file": obj_file.name,
        "vertex_label_file": str(vertex_label_file) if vertex_label_file else None,
        "jaw_type": "lower" if "lower" in obj_file.name else "upper",
        "num_vertex_labels": len(vertex_labels) if vertex_labels is not None else pd.NA,
        "num_gingiva_vertex_labels": np.sum(vertex_labels == 0) if vertex_labels is not None else pd.NA,
        "num_tooth_vertex_labels": np.sum(vertex_labels != 0) if vertex_labels is not None else pd.NA,
        "missing_teeth": get_missing_teeth(vertex_labels, vertex_label_file),
        "has_model_base": mesh_is_manifold,
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

df.info()
df.head()

# %% [markdown]
# ## Vertex label distribution

# %%
sns.histplot(df, x="num_vertex_labels", hue="jaw_type")
plt.show()

df.groupby("jaw_type")["num_vertex_labels"].describe()

# %%
sns.histplot(df, x="num_gingiva_vertex_labels", hue="jaw_type")
plt.show()

df.groupby("jaw_type")["num_gingiva_vertex_labels"].describe()

# %%
sns.histplot(df, x="num_tooth_vertex_labels", hue="jaw_type")
plt.show()

df.groupby("jaw_type")["num_tooth_vertex_labels"].describe()

# %%
# filter out rows where 'missing_teeth' is NaN/None
df_missing_teeth = df[df["missing_teeth"].notna()].copy()
# explode the list so each missing tooth gets its own row
df_missing_teeth = df_missing_teeth.explode("missing_teeth")
# reset the index to remove duplicate indices caused by exploding
df_missing_teeth = df_missing_teeth.reset_index(drop=True)

plt.figure(figsize=(12, 6))
sns.histplot(df_missing_teeth, x="missing_teeth", hue="jaw_type", binwidth=1, discrete=True)
plt.xticks(range(11, 49), rotation=45)
plt.show()

df.groupby("jaw_type")["missing_teeth"].describe()

# %% [markdown]
# ## Model / Mesh properties

# %%
# TODO: check for model base

sns.countplot(df, x="has_model_base", hue="jaw_type")
plt.show()

df.groupby("jaw_type")["has_model_base"].describe()

# %%
# check if obj mesh already has color/material attributes

df["mesh_has_material"].describe()

# %% [markdown]
# ## Potential dataset inconsistencies

# %%
# only lower or upper jaw in dataset

# get the set of unique patient IDs for each jaw type
ids_lower = set(df[df['jaw_type'] == 'lower']['id_patient'])
ids_upper = set(df[df['jaw_type'] == 'upper']['id_patient'])
print(f"ids_lower={len(ids_lower)}, ids_upper={len(ids_upper)}")

# identify the "missing" cases (only upper or lower jaw present in the dataset)
missing_ids_upper = list(ids_lower - ids_upper)
missing_ids_lower = list(ids_upper - ids_lower)
print(f"missing_ids_upper={missing_ids_upper}")
print(f"missing_ids_lower={missing_ids_lower}")
