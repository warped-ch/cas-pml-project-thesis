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
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import notebook_utils as nb_utils
import numpy as np
import seaborn as sns
import supervision as sv
import torch
import yaml

sys.path.append(str(Path.cwd().parent))
from src import inference_pipeline

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

best_model_chkpt_path = root_path / config.get("best_model_chkpt_path")
print(f"best_model_chkpt_path={best_model_chkpt_path}")

dataset_path_3d = root_path / config.get("dataset_path_3d")
print(f"dataset_path_3d={dataset_path_3d}")

temp_path = root_path / config.get("temp_path")
print(f"temp_path={temp_path}")

ip = inference_pipeline.InferencePipeline(
    chkpt_file=str(best_model_chkpt_path), config=config, device=device
)
print(f"model.class_names: {ip.model.class_names}")
print(f"model.model_config.resolution: {ip.model.model_config.resolution}")

# %%
test_split_path = root_path / "data" / "Teeth2D" / "test"

png_files = list(test_split_path.rglob("*.png"))
test_split_sample_ids = prefixes = {"_".join(f.stem.split("_")[:2]) for f in png_files}
print(f"test_split_sample_ids: {len(test_split_sample_ids)}")

# %%
colors = nb_utils.get_colors()
print(f"colors: {len(colors)}, {colors}")

class_ids = config["class_ids"]
sns.palplot(colors)
plt.title("FDI class_id color labels", fontsize=16, pad=20)
plt.xticks(range(len(colors)), class_ids)
plt.show()

colors_pv = nb_utils.convert_colors_pv(colors, config["class_ids"])
print(f"colors_pv: {len(colors_pv)}, {colors_pv}")
colors_sv = nb_utils.convert_colors_sv(colors)
print(f"colors_sv: {len(colors_sv)}, {colors_sv}")

# %%
test_sample = random.choice(list(test_split_sample_ids))
print(f"test_sample={test_sample}")

obj_file = str(next(dataset_path_3d.rglob(f"{test_sample}.obj")))
print(f"obj_file={obj_file}")

temp_path_sample = temp_path / Path(obj_file).stem
print(f"temp_path_sample={temp_path_sample}")
temp_path_sample.mkdir(parents=True, exist_ok=True)

mesh, vertex_labels, instances = ip.run_inference(obj_file)
for i, det in enumerate(ip.detections):
    cv2.imwrite(temp_path_sample / f"image_{i}.png", det.metadata["source_image"])
for i, mask in enumerate(ip.masks):
    cv2.imwrite(temp_path_sample / f"mask_{i}.png", mask)

annotated_images = []

mask_annotator = sv.MaskAnnotator(color=colors_sv)
label_annotator = sv.LabelAnnotator(
    color=colors_sv,
    text_color=sv.Color.BLACK,
    text_scale=0.4,
    text_padding=0,
    text_position=sv.Position.CENTER,
)

for i, det in enumerate(ip.detections):
    labels = [class_name.replace("class_", "") for class_name in det["class_name"]]
    print(f"det[{i}]: class_ids={det.class_id}, labels={labels}")

    annotated_img = mask_annotator.annotate(
        scene=det.metadata["source_image"].copy(), detections=det
    )
    annotated_img = label_annotator.annotate(annotated_img, det, labels)
    annotated_images.append(annotated_img)

views = np.array(config["2d_projection"]["views"])

nb_utils.plot_mesh(mesh, ip.vertex_labels_raw, colors_pv)
nb_utils.plot_mesh(mesh, vertex_labels, colors_pv)

nb_utils.plot_image_grid(
    images=annotated_images,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
)
