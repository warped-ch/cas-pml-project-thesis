# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: project-thesis (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data Preparation

# %% [markdown]
# ## Dataset: Teeth3DS+
#
# Download the following archives from: https://osf.io/xctdy
#
#   - 3DTeethSeg22_challenge_train_test_split.zip
#   - Teeth3DS_train_test_split.zip
#   - license.txt
#   - data_part_1.zip
#   - ...
#   - data_part_7.zip

# %%
import yaml
from pathlib import Path

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# %%
# load Teeth3DS+ dataset
# https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.datasets.Teeth3DS.html#torch_geometric.datasets.Teeth3DS

from torch_geometric.datasets import Teeth3DS

dataset_path = root_path / config.get("dataset_path")
print(f"dataset_path={dataset_path}")
dataset_path.mkdir(parents=True, exist_ok=True)

split = "3DTeethSeg22_challenge",  # Teeth3DS, 3DTeethSeg22_challenge, 3DTeethLand_challenge

train_dataset = Teeth3DS(
    root=dataset_path,
    split=split,
    train=True,
)

test_dataset = Teeth3DS(
    root=dataset_path,
    split=split,
    train=False,
)
