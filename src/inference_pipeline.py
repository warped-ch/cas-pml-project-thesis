from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pytorch3d.structures import Meshes
from rfdetr import RFDETRSegMedium

from . import file_io, view_projector


class InferencePipeline:
    def __init__(
        self,
        chkpt_file: str,
        config: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device

        self.view_projector = view_projector.ViewProjector(config=config, device=device)

        if Path(chkpt_file).suffix == ".pth":
            # TODO: check [WARNING] rf-detr - Model is not optimized for inference.
            # https://rfdetr.roboflow.com/latest/learn/run/segmentation/#run-on-an-image
            self.model = RFDETRSegMedium(pretrain_weights=chkpt_file, device=device)
        elif Path(chkpt_file).suffix == ".ckpt":
            # TODO: not working yet
            self.model = RFDETRSegMedium.from_checkpoint(path=chkpt_file, device=device)

        # TODO: temp stuff for debugging
        self.detections_raw = None
        self.detections = None
        self.mask = None

    # TODO: "class_1" instead of "class_0"
    # TODO: hack?
    def get_mask_value(self, class_id: int) -> int:
        class_name = self.model.class_names[class_id]
        mask_value = int(class_name.split("_")[1])
        return mask_value

    def run_inference(
        self, obj_file: str, threshold: float = 0.5
    ) -> tuple[Meshes, NDArray[np.uint8]]:
        self.detections_raw = None
        self.detections = None
        self.masks = None

        mesh = file_io.load_mesh_origin_aligned(obj_file, device=self.device)

        images = self.view_projector.render_2d_images_tensor(mesh)

        # tensor from render_2d_images_tensor is in (Batch, H, W, 3) format
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            print(
                f"run_inference: images.shape={images.shape}, images.dim={images.dim()}"
            )
            # convert from (H, W, C) to (C, H, W): permute(0, 3, 1, 2)
            images = images.permute(0, 3, 1, 2)
            print(
                f"run_inference: images.shape={images.shape}, images.dim={images.dim()}"
            )

        self.detections_raw = self.model.predict(
            images=list(images),
            threshold=threshold,
        )

        # Filter detections to keep only the best detection per class (highest confidence)
        # TODO: this will break "Teeth3DS+\raw\upper\SJDH33M1\SJDH33M1_upper.obj"
        self.detections = []
        for det_raw in self.detections_raw:
            # apply class agnostic NMS to the individual detections
            # NMS: https://github.com/roboflow/rf-detr/issues/187
            # TODO: consider using with_nmm?
            # https://supervision.roboflow.com/develop/detection/core/#supervision.detection.core.Detections.with_nmm
            det = det_raw.with_nms(threshold=0.5, class_agnostic=True)

            if len(det) == 0:
                self.detections.append(det)
                continue

            best_class_indices = []
            # loop the detections of one view image by class id
            for class_id in np.unique(det.class_id):
                # find indices for this specific class_id
                class_indices = np.where(det.class_id == class_id)[0]
                # find index of the detection with highest confidence for this class
                best_idx = class_indices[np.argmax(det.confidence[class_indices])]
                best_class_indices.append(best_idx)
            # keep only the best detection per class
            self.detections.append(det[np.array(best_class_indices)])

        # TODO: potential bug (removes gingiva, only teeth labels survive, rest is 0)
        self.masks = []
        # loop the filtered detections of all the view images
        for det_view in self.detections:
            _, h, w = det_view.mask.shape
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            # loop the filtered detections of one view image by class id
            for mask, class_id in zip(det_view.mask, det_view.class_id):
                mask_value = self.get_mask_value(class_id)
                combined_mask[mask] = mask_value
            self.masks.append(combined_mask)

        vertex_labels = self.view_projector.back_project_vertex_labels(
            mesh, np.stack(self.masks, axis=0)
        )

        return (mesh.cpu(), vertex_labels)
