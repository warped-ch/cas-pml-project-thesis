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
dataset_path_2d.mkdir(parents=True, exist_ok=True)
print(f"dataset_path_2d={dataset_path_2d}")

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# TODO: link this to "image_size" in config
model = RFDETRSegMedium()

output_dir = root_path / config["rf_train_output"] / timestamp

# Recommended configurations for different GPUs:
# https://rfdetr.roboflow.com/latest/learn/train/training-parameters/#understanding-batch-size
model.train(
    dataset_dir=str(dataset_path_2d),
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=output_dir,
)
