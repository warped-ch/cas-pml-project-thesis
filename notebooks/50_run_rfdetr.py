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

best_model_chkpt_path = (
    root_path
    / config.get("output_rf_detr_train")
    / "20260829_114546/Teeth2D/checkpoint_best_ema.pth"
)
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
mask_annotator = sv.MaskAnnotator(
    # color=sv.ColorPalette.from_matplotlib("viridis", len(np.unique(vertex_labels))),
    # TODO: custom colormap for good coloring? (viridis is good for lower, but bad for upper)
    # color=sv.ColorPalette.from_matplotlib("viridis", len(ip.model.class_names)),
)
for det in ip.detections:
    annotated_img = mask_annotator.annotate(
        scene=det.metadata["source_image"], detections=det
    )
    annotated_images.append(annotated_img)

label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER, text_padding=0)
for i, det in enumerate(ip.detections):
    labels = [
        ip.model.class_names[class_id].replace("class_", "")
        for class_id in det.class_id
    ]
    annotated_images[i] = sv.LabelAnnotator(
        text_position=sv.Position.CENTER, text_padding=0
    ).annotate(annotated_images[i], det, labels)

views = np.array(config["2d_projection"]["views"])

nb_utils.plot_mesh(mesh, vertex_labels)
nb_utils.plot_image_grid(
    images=annotated_images,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
)

# %%
from rfdetr import RFDETRSegMedium

model = RFDETRSegMedium(pretrain_weights=str(best_model_chkpt_path), device=device)

image = Path(
    r"C:\Development\cas_pml\project_thesis\data\Teeth2D_lower_has_model_base_false\test\ZM8PCSK6_lower_elev-60_azim0.png"
)
detections = model.predict(str(image), threshold=0.5)

annotated_image = mask_annotator.annotate(
    detections.metadata["source_image"], detections
)
cv2.imwrite(
    temp_out_path / f"mask_annotator1.png",
    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
)

labels = [
    model.class_names[class_id].replace("class_", "")
    for class_id in detections.class_id
]
annotated_image = sv.LabelAnnotator(
    text_position=sv.Position.CENTER, text_padding=0
).annotate(annotated_image, detections, labels)
cv2.imwrite(
    temp_out_path / f"mask_annotator2.png",
    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
)

