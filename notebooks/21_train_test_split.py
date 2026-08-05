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
# # Train/Test Split

# %%
from pathlib import Path

import yaml

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# %%
def load_official_splits(config) -> tuple[list[str], list[str]]:
    base_path = root_path / config["dataset_path_3d"] / "raw" / "Teeth3DS_train_test_split"

    train_files = ["training_lower.txt", "training_upper.txt"]
    test_files = ["testing_lower.txt", "testing_upper.txt"]

    def load_files(filenames: list[str]) -> list[str]:
        data = []
        for name in filenames:
            file = base_path / name
            content = file.read_text(encoding='utf-8').splitlines()
            data.extend([line for line in content if line.strip()])
        data.sort()
        return data

    train_split = load_files(train_files)
    test_split = load_files(test_files)

    return train_split, test_split


# %%
def save_splits(config, train_split: list[str], test_split: list[str]):
    with open(root_path / config["train_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_split))
    with open(root_path / config["test_split"], 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_split))


# %%
# use the official train/test splits for now

train_split, test_split = load_official_splits(config)
print(f"train_split={len(train_split)}")
print(f"test_split={len(test_split)}")

save_splits(config, train_split, test_split)
