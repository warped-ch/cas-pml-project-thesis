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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    return sorted(list(missing_teeth))


# %%
def load_sample(scan_file_path):
    vertex_label_file = scan_file_path.with_suffix(".json")
    # dataset contains samples from 3DTeethLand_challenge, skip those (no vertex label files available)
    if not vertex_label_file.exists():
        print(f"⚠️ vertex_label_file does not exist: '{vertex_label_file}'")
        return None

    json_data = {}
    if vertex_label_file:
        with open(vertex_label_file, "r") as json_file:
            json_data = json.load(json_file)

    vertex_labels = np.array(json_data["labels"]) if "labels" in json_data else None

    return {
        "id_patient": json_data.get("id_patient", None),
        "scan_file": scan_file_path.name,
        "vertex_label_file": str(vertex_label_file) if vertex_label_file else None,
        "jaw_type": "lower" if "lower" in scan_file_path.name else "upper",
        "num_vertex_labels": len(vertex_labels) if vertex_labels is not None else pd.NA,
        "num_gingiva_vertex_labels": np.sum(vertex_labels == 0) if vertex_labels is not None else pd.NA,
        "num_tooth_vertex_labels": np.sum(vertex_labels != 0) if vertex_labels is not None else pd.NA,
        "missing_teeth": get_missing_teeth(vertex_labels, vertex_label_file),
    }


# %%
# load the data

dataset_path = root_path / config.get("dataset_path")
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

scan_files = list(dataset_path.rglob("*.obj"))

# multiprocessing drops execution time from ~25s to ~23s
with ThreadPoolExecutor() as executor:
    data = list(
        tqdm(
            executor.map(load_sample, scan_files),
            total=len(scan_files),
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

# %%
# check if obj mesh already has color/material attributes

mesh_has_material = 0

for obj_file in tqdm(scan_files, desc="Checking mesh for material"):
    with open(obj_file, 'r', encoding='utf-8') as file:
        content = file.read()
        if "mtl" in content.lower():
            print(f"Mesh has material: {obj_file}")
            mesh_has_material += 1

print(f"mesh_has_material={mesh_has_material}")
