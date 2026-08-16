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
        self.detections = None
        self.mask = None

    # TODO: "class_1" instead of "class_0"
    # TODO: hack?
    def get_mask_value(self, class_id: int) -> int:
        class_name = self.model.class_names[class_id]
        mask_value = int(class_name.split("_")[1])
        return mask_value

    def run_inference(self, obj_file: str) -> tuple[Meshes, NDArray[np.uint8]]:
        self.detections = None
        self.masks = None

        mesh = file_io.load_mesh_origin_aligned(obj_file, device=self.device)

        images = self.view_projector.render_2d_images(mesh)

        # tensor from render_2d_images is in (Batch, H, W, C) format
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            print(f"run_inference: images.shape={images.shape}, images.dim={images.dim()}")
            # remove alpha channel: images[..., :3]
            # convert from (H, W, C) to (C, H, W): permute(0, 3, 1, 2)
            images = images[..., :3].permute(0, 3, 1, 2)
            print(f"run_inference: images.shape={images.shape}, images.dim={images.dim()}")

        self.detections = self.model.predict(
            images=list(images),
            threshold=0.5,
        )

        # TODO: avoid CPU roundtrip?
        self.masks = []
        for det in self.detections:
            _, h, w = det.mask.shape
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask, class_id in zip(det.mask, det.class_id):
                mask_value = self.get_mask_value(class_id)
                combined_mask[mask] = mask_value
            self.masks.append(combined_mask)

        vertex_labels = self.view_projector.back_project_vertex_labels(
            mesh, np.stack(self.masks, axis=0)
        )

        return (mesh.cpu(), vertex_labels)
