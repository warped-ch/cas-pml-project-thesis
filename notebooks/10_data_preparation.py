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
# # Data Preparation

# %%
import shutil
import sys
from pathlib import Path

import cv2
import fiftyone as fo
import pandas as pd
import torch
import yaml
from torch_geometric.datasets import Teeth3DS
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src import file_io, teeth3ds_utils, view_projector
from src.teeth2d_dataset_importer import Teeth2DDatasetImporter

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
dataset_path_2d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_2d={dataset_path_2d}")

dataset_path_3d = root_path / config.get("dataset_path_3d")
dataset_path_3d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_3d={dataset_path_3d}")

projections_path = root_path /config["data_path_projections"]
projections_path.mkdir(parents=True, exist_ok=True)
print(f"projections_path={projections_path}")

# %% [markdown]
# ## Dataset: Teeth3DS+
#
# - Download [Teeth3DS](https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.datasets.Teeth3DS.html) dataset from PyTorch Geometric.
# - Clean up leftovers from `3DTeethLand_challenge` split.

# %%
split = "3DTeethSeg22_challenge",  # Teeth3DS, 3DTeethSeg22_challenge, 3DTeethLand_challenge

train_dataset = Teeth3DS(
    root=dataset_path_3d,
    split=split,
    train=True,
)

test_dataset = Teeth3DS(
    root=dataset_path_3d,
    split=split,
    train=False,
)

# Teeth3DS+ dataset still contains samples from 3DTeethLand_challenge (no vertex label files available), remove those
obj_files = list(dataset_path_3d.rglob("*.obj"))
for obj_file in obj_files:
    vertex_label_file = obj_file.with_suffix(".json")
    if not vertex_label_file.exists():
        print(f"removing sample: '{vertex_label_file.parent}'")
        shutil.rmtree(obj_file.parent, ignore_errors=True)

# %% [markdown]
# ## Multi-View Projection Rendering
#
# - Render multi-view projections of the 3D models.
# - Save view and mask images.

# %%
shutil.rmtree(projections_path, ignore_errors=True)

images_path = projections_path / "images"
images_path.mkdir(parents=True, exist_ok=True)
print(f"images_path={images_path}")

masks_path = projections_path / "masks"
masks_path.mkdir(parents=True, exist_ok=True)
print(f"masks_path={masks_path}")

class_id_map = config["class_id_map"]

view_proj = view_projector.ViewProjector(config, device)

obj_files = list(dataset_path_3d.rglob("*.obj"))

for obj_file in tqdm(obj_files, desc="Rendering 2D views"):
    mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)
    vertex_labels = file_io.load_vertex_labels(obj_file.with_suffix(".json"), class_id_map)
    images, masks = view_proj.render_2d_views(mesh, vertex_labels)

    for image, mask, view in zip(images, masks, view_proj.views):
        elev = view[0]
        azim = view[1]

        image_file = images_path / f"{obj_file.stem}_elev{elev}_azim{azim}.png"
        image_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(image_file, image)

        mask_file = masks_path / f"{obj_file.stem}_elev{elev}_azim{azim}.png"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(mask_file, mask)

# %% [markdown]
# ## Dataset: Teeth2D
#
# - Create `Teeth2D` fiftyone dataset from the multi-view projection images and masks.
# - Create train/test split, update fiftyone dataset sample tags accordingly. 
# - Export `Teeth2D` in COCO dataset format.

# %%
dataset_importer = Teeth2DDatasetImporter(config=config, dataset_dir=str(projections_path))

fo_dataset = fo.Dataset.from_importer(
  name=str(dataset_path_2d.stem),
  dataset_importer=dataset_importer,
  overwrite=True)

fo_dataset.save()

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()


# %% [markdown]
# ### Train/Test split

# %%
def save_splits(config, train_split: list[str], test_split: list[str], val_split: list[str]):
    with open(root_path / config["train_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_split))
    with open(root_path / config["test_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_split))
    with open(root_path / config["val_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_split))

# use the official train/test splits for now

official_train_split, official_test_split = teeth3ds_utils.load_official_splits(config, root_path)
print(f"official_train_split: {len(official_train_split)} samples")
print(f"official_test_split: {len(official_test_split)} samples")

# train on full public train split (skip val for now), use test for evaluation
train_split = official_train_split
test_split = official_test_split
val_split = []
print(f"train_split: {len(train_split)}, test_split: {len(test_split)}, val_split: {len(val_split)}")

save_splits(config, train_split, test_split, val_split)

# %%
# update Teeth2D dataset sample tags:
#  - split tags: train, test, valid
#  - jaw tags: lower, upper
#  - has_model_base: true, false

dataset_path_2d = root_path / config["dataset_path_2d"]

# TODO: won't work until '20_data_exploration.ipynb' has been run...
df_meta = pd.read_csv(root_path / config["metadata"])

fo_dataset = fo.load_dataset(name=str(dataset_path_2d.stem))

tags = ["train", "test", "valid", "lower", "upper", "has_model_base_true", "has_model_base_false"]

# clear old split tags
fo_dataset.untag_samples(tags)

sample_split_prefixes = {
    "train": set(train_split),
    "test": set(test_split),
    "valid": set(val_split),
}

sample_has_model_base_prefixes = {
    "true": {Path(p).stem for p in df_meta.loc[df_meta["has_model_base"] == True, "obj_file"]},
    "false": {Path(p).stem for p in df_meta.loc[df_meta["has_model_base"] == False, "obj_file"]},
}

samle_tag_ids = {tag: [] for tag in tags}

ids, filepaths = fo_dataset.values(["id", "filepath"])
for id, filepath in tqdm(zip(ids, filepaths), total=len(ids), desc="Filtering for sample tags"):
    filename = Path(filepath).name

    # filter samples for dataset splits
    if any(filename.startswith(p) for p in sample_split_prefixes["train"]):
        samle_tag_ids["train"].append(id)
    elif any(filename.startswith(p) for p in sample_split_prefixes["test"]):
        samle_tag_ids["test"].append(id)
    elif any(filename.startswith(p) for p in sample_split_prefixes["valid"]):
        samle_tag_ids["valid"].append(id)
    else:
        print(f"⚠️ unknown dataset split for sample: {filename}")

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

# %%
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

    for has_model_base in ["has_model_base_true", "has_model_base_false"]:
        dataset_out_path = Path(f"{dataset_path_2d}_{jaw}_{has_model_base}")
        shutil.rmtree(dataset_out_path, ignore_errors=True)
        for split in ["train", "test", "valid"]:
            view = fo_dataset.match_tags([split, jaw, has_model_base], bool=True, all=True)
            export_split(view, dataset_out_path / split)
