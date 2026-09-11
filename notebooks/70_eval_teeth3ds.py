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
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from tqdm.auto import tqdm

sys.path.append(str(Path.cwd().parent))
from src import inference_pipeline
from src._3DTeethSeg_MICCAI_Challenges import evaluation

root_path = Path.cwd().parent
print(f"root_path={root_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# load config file

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_path_2d = root_path / config.get("dataset_path_2d")
print(f"dataset_path_2d={dataset_path_2d}")
dataset_path_3d = root_path / config.get("dataset_path_3d")
print(f"dataset_path_3d={dataset_path_3d}")

best_model_chkpt_path = root_path / config.get("best_model_chkpt_path")
print(f"best_model_chkpt_path={best_model_chkpt_path}")

out_path = root_path / config.get("output_path")
print(f"out_path={out_path}")
out_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Evaluate using the "3DTeethSeg Challenge MICCAI 2022" metrics
#
# - https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges

# %%
# run inference pipeline on test split and save metrics

# best_model_chkpt_path = root_path / config.get("output_rf_detr_train") / "20260909_191810/Teeth2D_upper" / "checkpoint_best_ema.pth"
# print(f"best_model_chkpt_path={best_model_chkpt_path}")
# dataset_path_2d = root_path / config.get("data_path") / "Teeth2D_upper"
# print(f"dataset_path_2d={dataset_path_2d}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

ip = inference_pipeline.InferencePipeline(
    chkpt_file=str(best_model_chkpt_path), config=config, device=device
)
print(f"model.class_names: {ip.model.class_names}")
print(f"model.model_config.resolution: {ip.model.model_config.resolution}")
# TODO: bake into inference pipeline
views = np.array(config["2d_projection"]["views"])
num_views = views.shape[0]
ip.model.inference(compile=True, batch_size=num_views)

test_split_path = dataset_path_2d / "test"
png_files = list(test_split_path.rglob("*.png"))
test_split_sample_ids = prefixes = {"_".join(f.stem.split("_")[:2]) for f in png_files}
print(f"test_split_sample_ids: {len(test_split_sample_ids)}")

predictions = []
for sample_id in tqdm(test_split_sample_ids, desc="Running 3DTeethSeg evaluation"):
    obj_file = str(next(dataset_path_3d.rglob(f"{sample_id}.obj")))

    try:
        mesh, vertex_labels, instances = ip.run_inference(obj_file)
    except Exception as e:
        print(f"💥 exception caught while processing {obj_file}: {str(e)}")

    parts = sample_id.split("_")
    id_patient = parts[0]
    jaw = parts[1]

    pred_label_dict = {
        "id_patient": id_patient,
        "jaw": jaw,
        "labels": vertex_labels.tolist() if vertex_labels is not None else None,
        "instances": instances.tolist() if instances is not None else None,
    }
    predictions.append(pred_label_dict)

metrics_dict = {
    "model_chkpt_path": str(best_model_chkpt_path.relative_to(root_path)),
    "dataset_path_2d": str(dataset_path_2d.relative_to(root_path)),
    "dataset_path_3d": str(dataset_path_3d.relative_to(root_path)),
    "predictions": predictions,
}

metrics_file = out_path / f"3DTeethSeg_eval_{timestamp}.json"
print(f"metrics_file={metrics_file}")
with open(metrics_file, "w") as f:
    json.dump(metrics_dict, f)

# %%
# read back the saved metrics and run 3DTeethSeg evaluation

# metrics_file = out_path / "3DTeethSeg_eval_20260908_120309.json"
with open(metrics_file, "r") as f:
    metrics_dict = json.load(f)

obj_files = list(dataset_path_3d.rglob("*.obj"))

TLA, TSA, TIR = [], [], []
for pred_label_dict in tqdm(metrics_dict["predictions"], desc="Evaluating predictions"):
    try:
        id_patient = pred_label_dict.get("id_patient")
        jaw = pred_label_dict.get("jaw")

        search_str = f"{id_patient}_{jaw}"
        obj_file = next((f for f in obj_files if search_str in f.name), None)
        gt_json_file = obj_file.with_suffix(".json")

        mesh = trimesh.load(obj_file, process=False)

        with open(gt_json_file, "r") as f:
            gt_label_dict = json.load(f)

        gt_label_dict["mesh_vertices"] = np.array(mesh.vertices)
        gt_label_dict["labels"] = np.array(gt_label_dict["labels"])
        gt_label_dict["instances"] = np.array(gt_label_dict["instances"])

        pred_label_dict["labels"] = np.array(pred_label_dict["labels"])
        pred_label_dict["instances"] = np.array(pred_label_dict["instances"])

        jaw_TLA, jaw_TSA, jaw_TIR = evaluation.calculate_metrics(
            gt_label_dict, pred_label_dict
        )
        TLA.append(math.exp(-jaw_TLA))
        TSA.append(jaw_TSA)
        TIR.append(jaw_TIR)
    except Exception as e:
        print(
            f"💥 exception caught while processing prediction {pred_label_dict}: {str(e)}"
        )
        TLA.append(0)
        TSA.append(0)
        TIR.append(0)
        continue

score = (np.mean(TSA) + np.mean(TLA) + np.mean(TIR)) / 3
print("TSA : {} +- {}".format(np.mean(TSA), np.std(TSA)))
print("TLA : {} +- {}".format(np.mean(TLA), np.std(TLA)))
print("TIR : {} +- {}".format(np.mean(TIR), np.std(TIR)))
print(" score : ", score)

# export metrics to /output/metrics.json
score_dict = {
    "global": score,
    "TSA": np.mean(TSA),
    "TLA": np.mean(TLA),
    "TIR": np.mean(TIR),
    "model_chkpt_path": str(best_model_chkpt_path.relative_to(root_path)),
    "dataset_path_2d": str(dataset_path_2d.relative_to(root_path)),
    "dataset_path_3d": str(dataset_path_3d.relative_to(root_path)),
}

eval_metrics_file = metrics_file.parent / f"{metrics_file.stem}_results.json"
with open(eval_metrics_file, "w") as f:
    json.dump(score_dict, f)
