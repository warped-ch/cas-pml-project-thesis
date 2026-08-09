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
# # Train a RF-DETR segmentation model on Teeth2D dataset

# %%
import json
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from rfdetr import RFDETRSegMedium

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
print(f"dataset_path_2d={dataset_path_2d}")

# %%
# TEMP create smaller datasets for testing

import fiftyone as fo

num_train_samples = 1000
num_test_samples = (int)(0.2*num_train_samples)

dataset_test_path = root_path / "data" / "Teeth2D_test"

shutil.rmtree(dataset_test_path, ignore_errors=True)

def export_split(view, path):
    print(f"exporting dataset split: {path}")
    view.export(
        dataset_type=fo.types.COCODetectionDataset,
        export_dir=str(path),
        labels_path="_annotations.coco.json",
        data_path=str(path),
        label_field="ground_truth_det",
        export_media=True,
        abs_paths=False,
        overwrite=True,
        # TODO: should use: tolerance=0, # Keeps every pixel boundary point
    )

dataset = fo.load_dataset("Teeth2D")

train_view = dataset.match_tags("train").limit(num_train_samples)
train_path = dataset_test_path / "train"
export_split(train_view, train_path)

test_view = dataset.match_tags("test").limit(num_test_samples)
test_path = dataset_test_path / "test"
export_split(test_view, test_path)

valid_path = dataset_test_path / "valid"
valid_path.mkdir(parents=True, exist_ok=True)
valid_dummy = {
    "images": [],
    "annotations": [],
    "categories": []
}
with open(valid_path / "_annotations.coco.json", "w", encoding="utf-8") as f:
    json.dump(valid_dummy, f)

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# TODO: link this to "image_size" in config
model = RFDETRSegMedium()

output_dir = root_path / config["output_rf_detr_train"] / timestamp

# Recommended configurations for different GPUs:
# https://rfdetr.roboflow.com/latest/learn/train/training-parameters/#understanding-batch-size
model.train(
    dataset_dir=str(dataset_path_2d),
    epochs=50,
    batch_size=4,
    grad_accum_steps=16,
    lr=5e-5,
    output_dir=output_dir,
)
