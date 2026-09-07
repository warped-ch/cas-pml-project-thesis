# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: project-thesis (3.12.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src import fdi_utils, mesh_utils, teeth3ds_utils

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# %%
# TODO: unit tests
# print(f"has_model_base: {mesh_utils.has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\lower\DNSRP767\DNSRP767_lower.obj"))}")
# print(f"has_model_base: {mesh_utils.has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\upper\DNSRP767\DNSRP767_upper.obj"))}")
# print(f"has_model_base: {mesh_utils.has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\lower\018XZVD6\018XZVD6_lower.obj"))}")
# print(f"has_model_base: {mesh_utils.has_model_base(Path(r"C:\Development\cas_pml\project_thesis\data\Teeth3DS+\raw\upper\018XZVD6\018XZVD6_upper.obj"))}")

# %%
def load_sample(
    obj_file: list[str], official_train_split: list[str], official_test_split: list[str]
):
    try:
        vertex_label_file = obj_file.with_suffix(".json")
        if not vertex_label_file.exists():
            print(f"⚠️ vertex_label_file does not exist: '{vertex_label_file}'")
            return None

        mesh_has_material = False
        with open(obj_file, "r", encoding="utf-8") as file:
            content = file.read()
            if "mtl" in content.lower():
                mesh_has_material = True
                print(f"Mesh has material: {obj_file}")

        json_data = {}
        if vertex_label_file:
            with open(vertex_label_file, "r") as json_file:
                json_data = json.load(json_file)

        vertex_labels = np.array(json_data["labels"]) if "labels" in json_data else None

        return {
            "id_sample": str(obj_file.stem),
            "id_patient": json_data.get("id_patient", None),
            "jaw": "lower" if "lower" in obj_file.name else "upper",
            "official_split": "train" if obj_file.stem in official_train_split else ("test" if obj_file.stem in official_test_split else None),
            "num_vertex_labels": len(vertex_labels) if vertex_labels is not None else pd.NA,
            "num_gingiva_vertex_labels": np.sum(vertex_labels == 0) if vertex_labels is not None else pd.NA,
            "num_tooth_vertex_labels": np.sum(vertex_labels != 0) if vertex_labels is not None else pd.NA,
            "missing_teeth": fdi_utils.get_missing_teeth(vertex_labels, vertex_label_file),
            "has_model_base": mesh_utils.has_model_base(obj_file),
            "mesh_has_material": mesh_has_material,
        }
    except Exception as e:
        print(f"💥 exception caught while processing {obj_file}: {str(e)}")
        return None


# %%
# load the data

dataset_path = root_path / config.get("dataset_path_3d")
print(f"dataset_path={dataset_path}")
assert dataset_path.is_dir(), f"'dataset_path' does not exist: {dataset_path}"

official_train_split, official_test_split = teeth3ds_utils.load_official_splits(
    config, root_path
)

obj_files = list(dataset_path.rglob("*.obj"))

data = Parallel(n_jobs=-1)(
    delayed(load_sample)(obj, official_train_split, official_test_split)
    for obj in tqdm(obj_files, desc="Processing samples")
)

# Remove None values from the list
original_len = len(data)
data = [d for d in data if d is not None]
if (removed := original_len - len(data)) > 0:
    print(f"⚠️ Removed {removed} samples due to processing errors.")

# %%
# create the DataFrame

df = pd.DataFrame(data)

# automatic type converion (automatically converts None/NaN into pd.NA for integer columns)
df = df.convert_dtypes()
df.to_csv(str(root_path / config.get("dataframe_eda")), index=False)

df.info()
df.head()

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

# number of samples per jaw
total_jaws = df.groupby("jaw").size().to_dict()
print(f"total_jaws={total_jaws}")
# count occurrences of each missing tooth
df_missing_teeth_counts = df_missing_teeth.groupby(["jaw", "missing_teeth"]).size().reset_index(name="abs_count")
# calculate percentage (number of patients missing this tooth / total number of patients)
df_missing_teeth_counts["percentage"] = df_missing_teeth_counts.apply(
    lambda row: (row["abs_count"] / total_jaws[row["jaw"]]) * 100, axis=1
)

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")
ax = sns.barplot(data=df_missing_teeth_counts, x="missing_teeth", y="percentage", hue="jaw")
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f')
ax.set(ylim=(0, 100))
plt.xticks(rotation=45)
plt.xlabel("FDI tooth number")
plt.ylabel("Percentage of missing teeth (%)")
plt.title("Missing Teeth (relative to jaw)")
plt.show()

df.groupby("jaw")["missing_teeth"].describe()

# %% [markdown]
# ## Model / Mesh properties

# %%
# balace of models with and without model base

ax = sns.countplot(df, x="has_model_base", hue="jaw")
for container in ax.containers:
    ax.bar_label(container)
plt.show()

g = sns.catplot(data=df, kind="count", x="has_model_base", hue="jaw", col="official_split")
for ax in g.axes.flat:
    for container in ax.containers:
        ax.bar_label(container)
plt.show()

df.groupby("jaw")["has_model_base"].describe()

# check consistency
has_base_mismatches = df.groupby("id_patient").filter(
    lambda x: x["has_model_base"].nunique() > 1
)
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
ids_lower = set(df[df["jaw"] == "lower"]["id_patient"])
ids_upper = set(df[df["jaw"] == "upper"]["id_patient"])
print(f"ids_lower={len(ids_lower)}, ids_upper={len(ids_upper)}")

# identify the "missing" cases (only upper or lower jaw present in the dataset)
missing_ids_upper = list(ids_lower - ids_upper)
missing_ids_lower = list(ids_upper - ids_lower)
print(f"missing_ids_upper={missing_ids_upper}")
print(f"missing_ids_lower={missing_ids_lower}")
