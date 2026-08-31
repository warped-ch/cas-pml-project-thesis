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
import random
import sys
from pathlib import Path

import cv2
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

dataset_path_3d = root_path / config.get("dataset_path_3d")
print(f"dataset_path_3d={dataset_path_3d}")

best_model_chkpt_path = root_path / config.get("best_model_chkpt_path")
print(f"best_model_chkpt_path={best_model_chkpt_path}")

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
custom_palette = nb_utils.get_color_palette()
sns.palplot(custom_palette)

def rgb_hex_to_bgr_hex(hex_str):
    hex_str = hex_str.lstrip('#')
    # Slice the string: RR=0:2, GG=2:4, BB=4:6
    # Reassemble as BB GG RR
    return f"#{hex_str[4:6]}{hex_str[2:4]}{hex_str[0:2]}"

bgr_custom_palette = [rgb_hex_to_bgr_hex(c) for c in custom_palette]

sv_custom_palette = sv.ColorPalette.from_hex(bgr_custom_palette)

# %%
test_sample = random.choice(list(test_split_sample_ids))
print(f"test_sample={test_sample}")

obj_file = str(next(dataset_path_3d.rglob(f"{test_sample}.obj")))
print(f"obj_file={obj_file}")

temp_out_path = dataset_path_3d.parent / "temp" / Path(obj_file).stem
print(f"temp_out_path={temp_out_path}")
temp_out_path.mkdir(parents=True, exist_ok=True)

mesh, vertex_labels = ip.run_inference(obj_file)
# TODO: contains "0" and "1"
print(f"class ids: {np.unique(vertex_labels)}")
for i, det in enumerate(ip.detections):
    cv2.imwrite(temp_out_path / f"image_{i}.png", det.metadata["source_image"])
for i, mask in enumerate(ip.masks):
    cv2.imwrite(temp_out_path / f"mask_{i}.png", mask)

annotated_images = []
mask_annotator = sv.MaskAnnotator(color=sv_custom_palette)
for det in ip.detections:
    color_dict_sv = {}
    for class_id in det.class_id:
        color_dict_sv[int(class_id)] = sv_custom_palette.by_idx(class_id).as_hex()
    print(f"color_dict_sv={color_dict_sv}")

    annotated_img = mask_annotator.annotate(
        scene=det.metadata["source_image"], detections=det
    )
    annotated_images.append(annotated_img)

label_annotator = sv.LabelAnnotator(
    color=sv_custom_palette, text_position=sv.Position.CENTER, text_padding=0
)
for i, det in enumerate(ip.detections):
    labels = [
        ip.model.class_names[class_id].replace("class_", "")
        for class_id in det.class_id
    ]
    annotated_images[i] = label_annotator.annotate(annotated_images[i], det, labels)

views = np.array(config["2d_projection"]["views"])

nb_utils.plot_mesh(mesh, vertex_labels)
nb_utils.plot_image_grid(
    images=annotated_images,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
)
