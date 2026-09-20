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

# %% [markdown]
# # Evaluate RF-DETR Teeth2D models

# %%
import json
from datetime import datetime
from pathlib import Path

import notebook_utils as nb_utils
import torch
from rfdetr import RFDETRSegMedium

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

config = nb_utils.load_config()

data_path = nb_utils.resolve_config_path("data_path", config)
print(f"data_path={data_path}")

train_output_path = nb_utils.resolve_config_path("output_rf_detr_train", config)
print(f"train_output_path={train_output_path}")

# %%
eval_list = [
    # Teeth2D
    ("20260915_185454", "Teeth2D", "Teeth2D"),
    ("20260912_010327", "Teeth2D", "Teeth2D"),
    ("20260909_191810", "Teeth2D", "Teeth2D"),
    # ("20260829_114546", "Teeth2D", "Teeth2D_custom"),
    # ("20260828_064008", "Teeth2D", "Teeth2D_custom"),
    # Teeth2D_lower
    ("20260915_185454", "Teeth2D_lower", "Teeth2D_lower"),
    ("20260912_010327", "Teeth2D_lower", "Teeth2D_lower"),
    ("20260909_191810", "Teeth2D_lower", "Teeth2D_lower"),
    ("20260909_035324", "Teeth2D_lower", "Teeth2D_lower"),
    # ("20260828_064008", "Teeth2D_lower", "Teeth2D_lower_custom"),
    # Teeth2D_upper
    ("20260915_185454", "Teeth2D_upper", "Teeth2D_upper"),
    ("20260912_010327", "Teeth2D_upper", "Teeth2D_upper"),
    ("20260909_191810", "Teeth2D_upper", "Teeth2D_upper"),
    ("20260909_035324", "Teeth2D_upper", "Teeth2D_upper"),
    # ("20260828_064008", "Teeth2D_upper", "Teeth2D_upper_custom"),
]

split = "test"

for train_out_folder, train_out_subfolder, dataset_name in eval_list:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_checkpoint = train_output_path / train_out_folder / train_out_subfolder / "checkpoint_best_ema.pth"
        print(f"model_checkpoint={model_checkpoint}")
        dataset_path = data_path / dataset_name
        print(f"dataset_path={dataset_path}")

        model = RFDETRSegMedium(pretrain_weights=str(model_checkpoint), device=device)
        print(f"model.class_names: {model.class_names}")
        print(f"model.model_config.resolution: {model.model_config.resolution}")

        batch_size = 8
        model.inference(compile=True, batch_size=batch_size)

        metrics = {}
        metrics = model.evaluate(
            dataset_dir=str(dataset_path),
            split=split,
            batch_size=batch_size,
            progress_bar="tqdm",
        )

        metrics_file = (
            model_checkpoint.parent
            / f"RF-DETR_eval_{split}_{model_checkpoint.stem}_{timestamp}.json"
        )
        print(f"metrics_file={metrics_file}")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
    except Exception as e:
        print(f"💥 exception caught while evaluation: {str(e)}")
