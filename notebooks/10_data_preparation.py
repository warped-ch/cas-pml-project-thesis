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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm

root_path = Path.cwd().parent
print(f"root_path={root_path}")


# %%
def process_file(src_file_path, dst_path):
    if zipfile.is_zipfile(src_file_path):
        with zipfile.ZipFile(src_file_path, "r") as archive:
            archive.extractall(dst_path)
    else:
        shutil.copy2(src_file_path, dst_path / src_file_path.name)


# %%
# ingest Teeth3DS+ dataset

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

# prevent "FileExistsError" for common subfolders
(dataset_dst_path / "lower").mkdir(parents=True, exist_ok=True)
(dataset_dst_path / "upper").mkdir(parents=True, exist_ok=True)

# multiprocessing drops execution time from ~1m to ~15s
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(process_file, dataset_src_path / f, dataset_dst_path): f
        for f in dataset_src_files
    }

    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Ingesting Dataset", unit="file"
    ):
        future.result()
