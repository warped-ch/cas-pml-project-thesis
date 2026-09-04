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

# %%
import json
from pathlib import Path

import torch
import yaml
from rfdetr import RFDETRSegMedium

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## Evaluate RF-DETR Teeth2D models

# %%
train_out_path = (
    root_path / config.get("output_rf_detr_train") / "20260828_064008/Teeth2D"
)
print(f"train_out_path={train_out_path}")

train_config_file = train_out_path / "training_config.json"
print(f"train_config_file={train_config_file}")
if train_config_file.is_file():
    with open(train_config_file, "r") as f:
        train_config = json.load(f)

    dataset_dir = Path(train_config["train_config"]["dataset_dir"])
else:
    print("⚠️ fallback to dataset_path_2d from own config")
    dataset_dir = root_path / config.get("dataset_path_2d")
print(f"dataset_dir={dataset_dir}")

split = "test"

pth_files = list(train_out_path.glob("*.pth"))
print(f"pth_files: {len(pth_files)}, {[pth_file.name for pth_file in pth_files]}")

for pth_file in pth_files:
    print(f"pth_file={pth_file}")

    model = RFDETRSegMedium(pretrain_weights=str(pth_file), device=device)
    print(f"model.class_names: {model.class_names}")
    print(f"model.model_config.resolution: {model.model_config.resolution}")

    metrics = {}
    metrics = model.evaluate(
        dataset_dir=str(dataset_dir),
        split=split,
        batch_size=64,
        progress_bar="tqdm",
    )

    metrics_file = train_out_path / f"eval_{split}_{pth_file.stem}.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"metrics_file={metrics_file}")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)
