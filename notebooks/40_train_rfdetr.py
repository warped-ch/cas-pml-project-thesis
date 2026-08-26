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
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# %%
datasets = [
    "Teeth2D_lower",
    # "Teeth2D_lower_has_model_base_false",
    # "Teeth2D_lower_has_model_base_true",
    # "Teeth2D_upper_has_model_base_false",
    # "Teeth2D_upper_has_model_base_true",
]
dataset_paths = [(root_path / "data" / ds) for ds in datasets]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for dataset_path in dataset_paths:
    # TODO: link this to "image_size" in config
    model = RFDETRSegMedium()

    output_dir = (
        root_path / config["output_rf_detr_train"] / timestamp / dataset_path.stem
    )
    print(f"output_dir={output_dir}")

    # Recommended configurations for different GPUs:
    # https://rfdetr.roboflow.com/latest/learn/train/training-parameters/#understanding-batch-size
    model.train(
        dataset_dir=str(dataset_path),
        output_dir=output_dir,
        epochs=200,
        batch_size=8,
        grad_accum_steps=2,
        lr=5e-5,
        aug_config=AUG_CONSERVATIVE,
        multi_scale=False,
        eval_interval=5,
        early_stopping=True,
        early_stopping_patience=10,  # Wait 10 epochs before stopping
        early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
        pin_memory=True,
        progress_bar="tqdm",
    )
