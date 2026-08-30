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

# %%
datasets = [
    # "Teeth2D_lower",
    # "Teeth2D_upper",
    "Teeth2D",
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

    # Notes on augmentation
    # https://rfdetr.roboflow.com/latest/learn/train/augmentations/#augmentations
    # - jaw context (models spatial anchor) is important (tooth 16 looks structurally identical to tooth 26), cannot rely on visual features alone
    # - geometric augmentation that preserves structure (mild rotation, scaling and elastic transforms)
    # - avoid flipping (e.g. horizontal flipping changes tooth 11 to tooth 21)
    # - avoid extreme crops (loosing jaw line curvature context)

    model.train(
        dataset_dir=str(dataset_path),
        output_dir=output_dir,
        epochs=200,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        aug_config={
            "transforms": [
                {"type": "Rotate", "limit": 10, "p": 0.3, "border_mode": 0}
            ]
        },
        save_dataset_grids=True,
        # TODO: speed up training by specifying "num_queries" according to classes in dataset?
        multi_scale=False,
        eval_interval=5,
        early_stopping=True,
        early_stopping_patience=10,  # Wait 10 epochs before stopping
        early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
        pin_memory=True,
        progress_bar="tqdm",
    )
