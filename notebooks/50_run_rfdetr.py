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
from pathlib import Path

import matplotlib.pyplot as plt
import supervision as sv
import yaml
from rfdetr import RFDETRSegMedium

root_path = Path.cwd().parent
print(f"root_path={root_path}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
print(f"dataset_path_2d={dataset_path_2d}")

# %%
weights_path = output_dir = root_path / config["rf_train_output"] / "20260807_192533" / "checkpoint_best_ema.pth"
print(f"weights_path={weights_path}")
model = RFDETRSegMedium(pretrain_weights=str(weights_path))

test_images = list((dataset_path_2d / "test").rglob("*.png"))
image_path = random.choice(test_images)
print(f"image_path={image_path}")

detections = model.predict(str(image_path), threshold=0.5)

annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)

#labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]
#annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

plt.imshow(annotated_image)
plt.title(str(image_path.name))
plt.axis('off')
plt.show()
