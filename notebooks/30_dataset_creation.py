# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown]
# # Dataset Creation
#
# - Create `Teeth2D` fiftyone dataset from the multi-view projection images and masks.
# - Update Teeth2D metadata sample tags.
# - Create custom train/valid/test splits.
# - Export `Teeth2D` dataset(s) in COCO format.

# %%
import shutil
import sys
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.random as four
import pandas as pd
import yaml
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

dataset_path_2d = root_path / config["dataset_path_2d"]

# TODO: won't work until '20_data_exploration.ipynb' has been run...
df_meta = pd.read_csv(root_path / config["metadata"])

fo_dataset = fo.load_dataset(name=str(dataset_path_2d.stem))

tags = ["lower", "upper", "has_model_base_true", "has_model_base_false"]

# clear old split tags
fo_dataset.untag_samples(tags)

sample_has_model_base_prefixes = {
    "true": {
        Path(p).stem for p in df_meta.loc[df_meta["has_model_base"] == True, "obj_file"]
    },
    "false": {
        Path(p).stem
        for p in df_meta.loc[df_meta["has_model_base"] == False, "obj_file"]
    },
}

samle_tag_ids = {tag: [] for tag in tags}

ids, filepaths = fo_dataset.values(["id", "filepath"])
for id, filepath in tqdm(
    zip(ids, filepaths), total=len(ids), desc="Filtering for sample tags"
):
    filename = Path(filepath).name

    # filter samples for jaw
    if "lower" in filename:
        samle_tag_ids["lower"].append(id)
    elif "upper" in filename:
        samle_tag_ids["upper"].append(id)
    else:
        print(f"⚠️ unknown jaw for sample: {filename}")

    # filter samples for model base
    if any(filename.startswith(p) for p in sample_has_model_base_prefixes["true"]):
        samle_tag_ids["has_model_base_true"].append(id)
    elif any(filename.startswith(p) for p in sample_has_model_base_prefixes["false"]):
        samle_tag_ids["has_model_base_false"].append(id)
    else:
        print(f"⚠️ 'has_model_base' does not exist for sample: {filename}")

for tag, ids in samle_tag_ids.items():
    view = fo_dataset.select(ids)
    print(f"view_{tag}: {len(view)} samples")
    view.tag_samples(tag)

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()

# %% [markdown]
# ### Train, Valid, Test split

# %%
# create sub-datasets splits for specialized model training

for jaw in ["lower", "upper"]:
    for has_model_base in ["has_model_base_false", "has_model_base_true"]:
        view = fo_dataset.match_tags([jaw, has_model_base], all=True)
        print(f"view: {len(view)}")

        # TODO: should keep all the view for patient_id together?
        four.random_split(view, {"train": 0.7, "valid": 0.15, "test": 0.15}, seed=42)


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

    for has_model_base in ["has_model_base_true", "has_model_base_false"]:
        dataset_out_path = Path(f"{dataset_path_2d}_{jaw}_{has_model_base}")
        shutil.rmtree(dataset_out_path, ignore_errors=True)
        for split in ["train", "test", "valid"]:
            view = fo_dataset.match_tags(
                [split, jaw, has_model_base], bool=True, all=True
            )
            export_split(view, dataset_out_path / split)
