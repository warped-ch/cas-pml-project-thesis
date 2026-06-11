# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: project-thesis (3.12.9)
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
import shutil
import zipfile
from pathlib import Path
from tqdm.auto import tqdm

root_path = Path.cwd().parent
print(f"root_path={root_path}")

dataset_src_path = Path(r"C:\Users\Andreas\Downloads\Teeth3DS+")
print(f"dataset_src_path={dataset_src_path}")
dataset_src_files = [
    "3DTeethSeg22_challenge_train_test_split.zip",
    "Teeth3DS_train_test_split.zip",
    "license.txt",
    "data_part_1.zip",
    "data_part_2.zip",
    "data_part_3.zip",
    "data_part_4.zip",
    "data_part_5.zip",
    "data_part_6.zip",
    "data_part_7.zip",
]

dataset_dst_path = root_path / "dataset" / "Teeth3DS+"
print(f"dataset_dst_path={dataset_dst_path}")
dataset_dst_path.mkdir(parents=True, exist_ok=True)

with tqdm(dataset_src_files) as pbar:
    for f in pbar:
        src_file = dataset_src_path / f
        assert src_file.is_file(), f"'src_file' does not exist: {src_file}"

        pbar.set_description(f"Processing {src_file.name}")

        if zipfile.is_zipfile(src_file):
            with zipfile.ZipFile(src_file, "r") as archive:
                archive.extractall(dataset_dst_path)
        else:
            shutil.copy2(src_file, dataset_dst_path / src_file.name)
