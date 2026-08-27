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
# # Dataset Creation
#
# - Create `Teeth2D` fiftyone dataset from the multi-view projection images and masks.
# - Update Teeth2D metadata sample tags.
# - Create custom train/valid/test splits.
# - Export `Teeth2D` dataset(s) in COCO format.

# %%
import ast
import shutil
import sys
from pathlib import Path

import fiftyone as fo
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src.teeth2d_dataset_importer import Teeth2DDatasetImporter

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
dataset_path_2d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_2d={dataset_path_2d}")

projections_path = root_path / config["data_path_projections"]
projections_path.mkdir(parents=True, exist_ok=True)
print(f"projections_path={projections_path}")

df_eda = pd.read_csv(
    root_path / config["dataframe_eda"], converters={"missing_teeth": ast.literal_eval}
)

# %%
dataset_importer = Teeth2DDatasetImporter(
    config=config, dataset_dir=str(projections_path)
)

fo_dataset = fo.Dataset.from_importer(
    name=str(dataset_path_2d.stem), dataset_importer=dataset_importer, overwrite=True
)

fo_dataset.save()

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()

# %% [markdown]
# ### Metadata tags

# %%
# update Teeth2D metadata sample tags:
#  - jaw tags: lower, upper
#  - has_model_base: true, false

fo_dataset = fo.load_dataset(name=str(dataset_path_2d.stem))

tags = ["lower", "upper", "has_model_base_true", "has_model_base_false"]

# clear old metadata tags
fo_dataset.untag_samples(tags)

sample_ids_has_model_base = {
    "true": {
        Path(p).stem for p in df_eda.loc[df_eda["has_model_base"] == True, "id_sample"]
    },
    "false": {
        Path(p).stem for p in df_eda.loc[df_eda["has_model_base"] == False, "id_sample"]
    },
}

sample_tag_ids = {tag: [] for tag in tags}

ids, filepaths = fo_dataset.values(["id", "filepath"])
for id, filepath in tqdm(
    zip(ids, filepaths), total=len(ids), desc="Filtering for sample tags"
):
    filename = Path(filepath).name

    # filter samples for jaw
    if "lower" in filename:
        sample_tag_ids["lower"].append(id)
    elif "upper" in filename:
        sample_tag_ids["upper"].append(id)
    else:
        print(f"⚠️ unknown jaw for sample: {filename}")

    # filter samples for model base
    if any(filename.startswith(p) for p in sample_ids_has_model_base["true"]):
        sample_tag_ids["has_model_base_true"].append(id)
    elif any(filename.startswith(p) for p in sample_ids_has_model_base["false"]):
        sample_tag_ids["has_model_base_false"].append(id)
    else:
        print(f"⚠️ 'has_model_base' does not exist for sample: {filename}")

for tag, ids in sample_tag_ids.items():
    view = fo_dataset.select(ids)
    print(f"view_{tag}: {len(view)} samples")
    view.tag_samples(tag)

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()


# %% [markdown]
# ### Train, Valid, Test split

# %%
def split_stratified(
    df,
    ids_to_split=None,
    stratify_col="jaw",
    test_size=0.15,
    val_size=0.15,
    random_state=42,
):
    """
    Splits a DataFrame or a specific list of Sample IDs into Train, Valid, and Test sets, stratified by a specific column.
    """
    # If IDs are provided, filter the dataframe; otherwise, use the whole dataframe
    if ids_to_split is not None:
        df_subset = df[df["id_sample"].isin(ids_to_split)].copy()
    else:
        df_subset = df.copy()

    # First Split: Separate Test from the rest (Train + Valid)
    train_val_df, test_df = train_test_split(
        df_subset,
        test_size=test_size,
        stratify=df_subset[stratify_col],
        random_state=random_state,
    )

    # Second Split: Separate Train and Val
    # Ratio = 0.15 / (1 - 0.15) = 0.15 / 0.85 ≈ 0.176
    adjusted_val_size = val_size / (1 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[stratify_col],
        random_state=random_state,
    )

    return (
        train_df["id_sample"].tolist(),
        val_df["id_sample"].tolist(),
        test_df["id_sample"].tolist(),
    )


# %%
unique_samples = df_eda["id_sample"].unique()
print(f"unique_samples: {len(unique_samples)}")

df_remaining = df_eda.copy()
split_sample_ids = {
    "train": [],
    "valid": [],
    "test": [],
}

# missing_teeth: []
# add all samples where all teeth are present to train set (rare samples)
mask_all_teeth = df_remaining["missing_teeth"].apply(len) == 0
sample_ids_all_teeth = df_remaining.loc[mask_all_teeth, "id_sample"]
print(f"sample_ids_all_teeth: {len(sample_ids_all_teeth)}")
df_remaining = df_remaining[~mask_all_teeth].copy()
print(f"df_remaining: {len(df_remaining)}")
split_sample_ids["train"] += sample_ids_all_teeth.to_list()

# has_model_base: false
# split models without base separately to ensure balanced representation of this rarer subset (approx. 1:2 ratio to models with base).
mask_has_model_base = df_remaining["has_model_base"] == True
sample_ids_no_base = df_remaining.loc[~mask_has_model_base, "id_sample"].to_list()
print(f"sample_ids_no_base: {len(sample_ids_no_base)}")
train_ids, valid_ids, test_ids = split_stratified(df_remaining, sample_ids_no_base)
print(
    f"train_ids: {len(train_ids)}, valid_ids: {len(valid_ids)}, test_ids: {len(test_ids)}"
)
df_remaining = df_remaining[mask_has_model_base].copy()
print(f"df_remaining: {len(df_remaining)}")
split_sample_ids["train"].extend(train_ids)
split_sample_ids["valid"].extend(valid_ids)
split_sample_ids["test"].extend(test_ids)

# split remaining dataframe
train_ids, valid_ids, test_ids = split_stratified(df_remaining)
print(
    f"train_ids: {len(train_ids)}, valid_ids: {len(valid_ids)}, test_ids: {len(test_ids)}"
)
split_sample_ids["train"].extend(train_ids)
split_sample_ids["valid"].extend(valid_ids)
split_sample_ids["test"].extend(test_ids)

total_samples = sum(len(v) for v in split_sample_ids.values())
for k, v in split_sample_ids.items():
    count = len(v)
    percent = (count / total_samples) * 100 if total_samples > 0 else 0
    print(f"{k}: {count} samples ({percent:2.1f}%)")

# %%
# add split tags (train, valid, test) to Teeth2D dataset

fo_dataset = fo.load_dataset(name=str(dataset_path_2d.stem))

tags = ["train", "valid", "test"]

# clear old split tags
fo_dataset.untag_samples(tags)

sample_tag_ids = {tag: [] for tag in tags}

ids, filepaths = fo_dataset.values(["id", "filepath"])
for id, filepath in tqdm(
    zip(ids, filepaths), total=len(ids), desc="Filtering for sample tags"
):
    filename = Path(filepath).name

    # filter samples for split
    if any(filename.startswith(p) for p in split_sample_ids["train"]):
        sample_tag_ids["train"].append(id)
    elif any(filename.startswith(p) for p in split_sample_ids["valid"]):
        sample_tag_ids["valid"].append(id)
    elif any(filename.startswith(p) for p in split_sample_ids["test"]):
        sample_tag_ids["test"].append(id)
    else:
        print(f"⚠️ 'split' does not exist for sample: {filename}")

for tag, ids in sample_tag_ids.items():
    view = fo_dataset.select(ids)
    print(f"view_{tag}: {len(view)} samples")
    view.tag_samples(tag)

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()


# %% [markdown]
# ### Export Teeth2D dataset

# %%
def export_split(view, path):
    print(f"exporting dataset split: {path}")
    view.export(
        dataset_type=fo.types.COCODetectionDataset,
        export_dir=str(path),
        labels_path="_annotations.coco.json",
        data_path=str(path),
        label_field="ground_truth_det",
        # export_media="symlink" might save some disk space but won't speed things up
        export_media=True,
        abs_paths=False,
        overwrite=True,
        # TODO: should use: tolerance=0, # Keeps every pixel boundary point
    )


# %%
# export Teeth2D COCO dataset train/test/val splits

# consider RF-DETR dataset format requirements:
# https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/#dataset-formats

shutil.rmtree(dataset_path_2d, ignore_errors=True)

for split in ["train", "test", "valid"]:
    view = fo_dataset.match_tags(split)
    export_split(view, dataset_path_2d / split)

# %%
# export Teeth2D sub-datasets for specialized model training

for jaw in ["lower", "upper"]:
    dataset_out_path = Path(f"{dataset_path_2d}_{jaw}")
    shutil.rmtree(dataset_out_path, ignore_errors=True)
    for split in ["train", "test", "valid"]:
        view = fo_dataset.match_tags([split, jaw], bool=True, all=True)
        export_split(view, dataset_out_path / split)
