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
# ## Evaluate on Teeth2D dataset

# %%
best_model_chkpt_path = (
    root_path
    / config.get("output_rf_detr_train")
    / "20260828_064008/Teeth2D/checkpoint_best_ema.pth"
)
print(f"best_model_chkpt_path={best_model_chkpt_path}")

dataset_path_2d = root_path / "data" / "Teeth2D"
print(f"dataset_path_2d={dataset_path_2d}")

split = "test"

metrics_file = (
    root_path
    / config["output_rf_detr_test"]
    / f"{best_model_chkpt_path.parent.parent.stem}_{best_model_chkpt_path.parent.stem}_{split}.json"
)
metrics_file.parent.mkdir(parents=True, exist_ok=True)
print(f"metrics_file={metrics_file}")

model = RFDETRSegMedium(pretrain_weights=best_model_chkpt_path, device=device)
print(f"model.class_names: {model.class_names}")
print(f"model.model_config.resolution: {model.model_config.resolution}")

metrics = model.evaluate(
    dataset_dir=str(dataset_path_2d),
    split=split,
    batch_size=64,
)
with open(metrics_file, "w") as f:
    json.dump(metrics, f)
print("Evaluation Metrics:", metrics)
