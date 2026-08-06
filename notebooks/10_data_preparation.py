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
import torch
import yaml
from torch_geometric.datasets import Teeth3DS
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src import file_io, view_projector
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

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()


# %% [markdown]
# ### Train/Test split

# %%
def load_official_splits(config) -> tuple[list[str], list[str]]:
    base_path = root_path / config["dataset_path_3d"] / "raw" / "Teeth3DS_train_test_split"

    train_files = ["training_lower.txt", "training_upper.txt"]
    test_files = ["testing_lower.txt", "testing_upper.txt"]

    def load_files(filenames: list[str]) -> list[str]:
        data = []
        for name in filenames:
            file = base_path / name
            content = file.read_text(encoding='utf-8').splitlines()
            data.extend([line for line in content if line.strip()])
        data.sort()
        return data

    train_split = load_files(train_files)
    test_split = load_files(test_files)

    return train_split, test_split

def save_splits(config, train_split: list[str], test_split: list[str]):
    with open(root_path / config["train_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_split))
    with open(root_path / config["test_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_split))


# %%
# use the official train/test splits for now

train_split, test_split = load_official_splits(config)
print(f"train_split={len(train_split)}")
print(f"test_split={len(test_split)}")

save_splits(config, train_split, test_split)

# %%
# update Teeth2D dataset with train/test split tags

dataset_path_2d = root_path / config["dataset_path_2d"]

fo_dataset = fo.load_dataset(name=str(dataset_path_2d.stem))

# clear old split tags
fo_dataset.untag_samples(["train", "test"])

ids, filepaths = fo_dataset.values(["id", "filepath"])

train_sample_ids = [
    sample_id for sample_id, filepath in zip(ids, filepaths)
    if any(Path(filepath).name.startswith(prefix) for prefix in set(train_split))
]
train_view = fo_dataset.select(train_sample_ids)
train_view.tag_samples("train")
print(f"Tagged 'train' samples: {len(train_view)}")

test_sample_ids = [
    sample_id for sample_id, filepath in zip(ids, filepaths)
    if any(Path(filepath).name.startswith(prefix) for prefix in set(test_split))
]
test_view = fo_dataset.select(test_sample_ids)
test_view.tag_samples("test")
print(f"Tagged 'test' samples: {len(test_view)}")

fo_dataset.save()

session = fo.launch_app(fo_dataset, auto=False)
session.open_tab()

# %% [markdown]
# ### Export `Teeth2D`

# %%
# export Teeth2D COCO dataset train/test splits

# consider RF-DETR dataset format requirements:
# https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/#dataset-formats

shutil.rmtree(dataset_path_2d, ignore_errors=True)

label_file_name = "_annotations.coco.json"

train_path = dataset_path_2d / "train"
train_view.export(
    dataset_type=fo.types.COCODetectionDataset,
    export_dir=str(train_path),
    labels_path=label_file_name,
    data_path=str(train_path),
    label_field="ground_truth_det",
    export_media=True,
    abs_paths=False,
    overwrite=True,
    # TODO: should use: tolerance=0, # Keeps every pixel boundary point
)

test_path = dataset_path_2d / "test"
test_view.export(
    dataset_type=fo.types.COCODetectionDataset,
    export_dir=str(test_path),
    labels_path=label_file_name,
    data_path=str(test_path),
    label_field="ground_truth_det",
    export_media=True,
    abs_paths=False,
    overwrite=True,
    # TODO: should use: tolerance=0, # Keeps every pixel boundary point
)
