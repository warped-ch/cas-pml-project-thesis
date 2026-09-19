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
import ast
import sys
from pathlib import Path

import cv2
import notebook_utils as nb_utils
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

sys.path.append(str(Path.cwd().parent))
from src import file_io, view_projector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

config = nb_utils.load_config()

dataset_path_3d = nb_utils.resolve_config_path("dataset_path_3d", config)
print(f"dataset_path_3d={dataset_path_3d}")

temp_path = nb_utils.resolve_config_path("temp_path", config)
print(f"temp_path={temp_path}")

colors = nb_utils.get_colors()
colors_pv = nb_utils.convert_colors_pv(colors, config["class_ids"])

# %%
df_eda_path = nb_utils.resolve_config_path("dataframe_eda", config)
print(f"df_eda_path={df_eda_path}")
df_eda = pd.read_csv(df_eda_path, converters={"missing_teeth": ast.literal_eval})
mask_all_teeth = df_eda["missing_teeth"].apply(len) == 0
sample_ids_all_teeth = df_eda.loc[mask_all_teeth, "id_sample"].tolist()

# %%
samples = [
  r"raw\lower\0EJBIPTC\0EJBIPTC_lower.obj",
  r"raw\upper\0EJBIPTC\0EJBIPTC_upper.obj",
  r"raw\lower\0U1LI1CB\0U1LI1CB_lower.obj",
  r"raw\upper\0U1LI1CB\0U1LI1CB_upper.obj",
]

for sample in samples:
    obj_file = dataset_path_3d / sample

    mesh = file_io.load_mesh_origin_aligned(obj_file, device=device)

    vertex_labels = file_io.load_vertex_labels(obj_file.with_suffix(".json"))

    nb_utils.plot_mesh(mesh, None, colors_pv)
    #nb_utils.plot_mesh(mesh, vertex_labels, colors_pv)
