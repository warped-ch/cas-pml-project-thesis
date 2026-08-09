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
import sys
from pathlib import Path

import notebook_utils as nb_utils
import numpy as np
import supervision as sv
import yaml
from rfdetr import RFDETRSegMedium

sys.path.append(str(Path.cwd().parent))
from src import file_io

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
print(f"dataset_path_2d={dataset_path_2d}")

# %%
best_model_chkpt_path = output_dir = root_path / config["best_model_chkpt_path"]
print(f"best_model_chkpt_path={best_model_chkpt_path}")
model = RFDETRSegMedium(pretrain_weights=str(best_model_chkpt_path))
print(f"model.class_names: {model.class_names}")
model_class_ids = {idx: name for idx, name in enumerate(model.class_names)}
print(f"model_class_ids: {model_class_ids}")

test_split_file = root_path / config["test_split"]
print(f"test_split_file={test_split_file}")
test_sample = file_io.read_random_line_from_file(str(test_split_file))
print(f"test_sample={test_sample}")

test_path = dataset_path_2d / "test"
test_image_paths = list(test_path.glob(f"{test_sample}*.png"))
print(f"test_image_paths={test_image_paths}")

detections = model.predict(
  [str(p) for p in test_image_paths],
  threshold=0.5
)

annotated_images = []
for det in detections:
    annotated_img = sv.MaskAnnotator().annotate(det.metadata["source_image"], det)
    annotated_images.append(annotated_img)

views = np.array(config["2d_projection"]["views"])
nb_utils.plot_image_grid(
    images=annotated_images,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views]
)
