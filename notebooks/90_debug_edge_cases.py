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

# %% [markdown]
# # Debugging edge cases

# %%
import sys
from pathlib import Path

import cv2
import notebook_utils as nb_utils
import numpy as np
import supervision as sv
import torch

sys.path.append(str(Path.cwd().parent))
from src import inference_pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

config = nb_utils.load_config()

best_model_chkpt_path = nb_utils.resolve_config_path("best_model_chkpt_path", config)
print(f"best_model_chkpt_path={best_model_chkpt_path}")

dataset_path_3d = nb_utils.resolve_config_path("dataset_path_3d", config)
print(f"dataset_path_3d={dataset_path_3d}")

temp_path = nb_utils.resolve_config_path("temp_path", config)
print(f"temp_path={temp_path}")

views = np.array(config["2d_projection"]["views"])

ip = inference_pipeline.InferencePipeline(
    chkpt_file=str(best_model_chkpt_path), config=config, device=device
)
print(f"model.class_names: {ip.model.class_names}")
print(f"model.model_config.resolution: {ip.model.model_config.resolution}")

# %%
colors = nb_utils.get_colors()
print(f"colors: {len(colors)}, {colors}")
colors_pv = nb_utils.convert_colors_pv(colors, config["class_ids"])
print(f"colors_pv: {len(colors_pv)}, {colors_pv}")
colors_sv = nb_utils.convert_colors_sv(colors)
print(f"colors_sv: {len(colors_sv)}, {colors_sv}")

nb_utils.plot_colors_fdi(colors, config["class_ids"])

# %%
# misprediction on 33, 34 / post processing leaves 33 completely enclosed by 34
obj_file = dataset_path_3d / "raw/lower/01KRDUKX/01KRDUKX_lower.obj"
# only two peaks visible for 14 / post processing relabels parts of it to 15
#obj_file = dataset_path_3d / "raw/upper/SJDH33M1/SJDH33M1_upper.obj"
# misprediction on 45, 46 / post processing failing on 46 (still some largest component leftover of mispredicted 47?)
#obj_file = dataset_path_3d / "raw/lower/01FAYE3T/01FAYE3T_lower.obj"
# misprediction on 15, 16 / postprocessing fails
#obj_file = dataset_path_3d / "raw/upper/CZ0XDR02/CZ0XDR02_upper.obj"
# exception in post processing (fixed)
#obj_file = dataset_path_3d / "raw/lower/J1A5FRQN/J1A5FRQN_lower.obj"
# TODO
#obj_file = dataset_path_3d / "raw/lower/AD8EQEUR/AD8EQEUR_lower.obj"
#obj_file = dataset_path_3d / "raw/upper/276N3YRC/276N3YRC_upper.obj"
#obj_file = dataset_path_3d / "raw/lower/3Y8A14TI/3Y8A14TI_lower.obj"
# complete mess up, also mixing lower/upper?
#obj_file = dataset_path_3d / "raw/upper/ZB8NU437/ZB8NU437_upper.obj"
# mixup on 14, 15, 16
#obj_file = dataset_path_3d / r"raw\upper\RMZC48A0\RMZC48A0_upper.obj"
# mixup on 41, 42
#obj_file = dataset_path_3d / r"raw\lower\ZM8PCSK6\ZM8PCSK6_lower.obj"
# mixup on 24, 25, 26
#obj_file = dataset_path_3d / r"raw\upper\mccarthy\mccarthy_upper.obj"
# mixup on 35, 36
#obj_file = dataset_path_3d / r"raw\lower\YJXIBVIM\YJXIBVIM_lower.obj"
# 35 labeled as 36
#obj_file = dataset_path_3d / r"raw\lower\0NH6X4SS\0NH6X4SS_lower.obj"
# fdi number mixup
obj_file = dataset_path_3d / r"raw\upper\KSHNN3DV\KSHNN3DV_upper.obj"
# double fdi labels on different teeth
obj_file = dataset_path_3d / r"raw\lower\KAHYFGOY\KAHYFGOY_lower.obj"

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

nb_utils.plot_mesh(mesh, ip.vertex_labels_raw, colors_pv)
nb_utils.plot_mesh(mesh, vertex_labels, colors_pv)

nb_utils.plot_image_grid(
    images=annotated_images,
    titles=[f"elevation={view[0]}, azimuth={view[1]}" for view in views],
)

# %%
# log individual per class detections
for i, det in enumerate(ip.detections):
    print(
        f"det[{i}]: class_ids={det.class_id}, fdi_labels={[class_name.replace('class_', '') for class_name in det['class_name']]}"
    )

    for class_id in det.class_id:
        det_filter = det[det.class_id == class_id]
        class_name = det_filter["class_name"]
        if len(class_name) > 1:
            print(f"⚠️ multiple objects for same class: class_name={class_name}")

        labels = [
            f"{name} {conf:.2f}"
            for name, conf in zip(det_filter["class_name"], det_filter.confidence)
        ]

        annotated_img = mask_annotator.annotate(
            scene=det.metadata["source_image"].copy(), detections=det_filter
        )
        annotated_img = label_annotator.annotate(
            scene=annotated_img, detections=det_filter, labels=labels
        )

        annotated_img_file_name = f"annotated_elevation_{views[i][0]}_azimuth_{views[i][1]}_{class_name[0]}.png"
        # allows better file ordering
        annotated_img_file_name = annotated_img_file_name.replace("-", "neg")
        cv2.imwrite(
            temp_path_sample / annotated_img_file_name,
            annotated_img,
        )
