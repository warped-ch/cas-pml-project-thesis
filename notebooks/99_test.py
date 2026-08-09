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
import torch
import yaml
import numpy as np
import notebook_utils as nb_utils

sys.path.append(str(Path.cwd().parent))
from src import file_io, inference_pipeline

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

best_model_chkpt_path = output_dir = root_path / config["best_model_chkpt_path"]
print(f"best_model_chkpt_path={best_model_chkpt_path}")

ip = inference_pipeline.InferencePipeline(
  chkpt_file=str(best_model_chkpt_path),
  config=config,
  device=device
)

# %%
dataset_path_3d = root_path / config.get("dataset_path_3d")
print(f"dataset_path_3d={dataset_path_3d}")

test_split_file = root_path / config["test_split"]
print(f"test_split_file={test_split_file}")
test_sample = file_io.read_random_line_from_file(str(test_split_file))
print(f"test_sample={test_sample}")

obj_file = str(next(dataset_path_3d.rglob(f"{test_sample}.obj")))
print(f"obj_file={obj_file}")

temp_out_path = dataset_path_3d.parent / "temp" / Path(obj_file).stem
print(f"temp_out_path={temp_out_path}")
temp_out_path.mkdir(parents=True, exist_ok=True)

mesh, vertex_labels = ip.run_inference(obj_file, temp_out_path)
# TODO: contains "0" and "1"
print(f"class ids: {np.unique(vertex_labels)}")

nb_utils.plot_mesh(mesh, vertex_labels)
