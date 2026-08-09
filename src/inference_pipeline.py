from typing import Any

import cv2
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

        # TODO: check [WARNING] rf-detr - Model is not optimized for inference.
        # https://rfdetr.roboflow.com/latest/learn/run/segmentation/#run-on-an-image
        self.model = RFDETRSegMedium(pretrain_weights=chkpt_file, device=device)
        print(f"model.class_names: {self.model.class_names}")

        self.reverse_class_id_lookup = np.arange(256, dtype=np.uint8)
        for old_id, new_id in self.config["class_id_map"].items():
            self.reverse_class_id_lookup[new_id] = old_id

    # TODO: "class_1" instead of "class_0"
    # TODO: hack?
    def get_mask_value(self, class_id: int) -> int:
        class_name = self.model.class_names[class_id]
        mask_value = int(class_name.split("_")[1])
        return mask_value

    def run_inference(
        self, obj_file: str, temp_out_path: str | None = None
    ) -> tuple[Meshes, NDArray[np.uint8]]:
        mesh = file_io.load_mesh_origin_aligned(obj_file, device=self.device)

        images = self.view_projector.render_2d_images(mesh)

        # tensor from render_2d_images is in (Batch, H, W, C) format
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            # remove alpha channel: images[..., :3]
            # convert from (H, W, C) to (C, H, W): permute(0, 3, 1, 2)
            images = images[..., :3].permute(0, 3, 1, 2)

        detections = self.model.predict(
            images=list(images),
            threshold=0.5,
        )

        # TODO: avoid CPU roundtrip?
        masks = []
        for det in detections:
            _, h, w = det.mask.shape
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for mask, class_id in zip(det.mask, det.class_id):
                mask_value = self.get_mask_value(class_id)
                combined_mask[mask] = mask_value
            masks.append(combined_mask)
        if temp_out_path:
            for i, mask in enumerate(masks):
                cv2.imwrite(temp_out_path / f"mask_{i}.png", mask)

        vertex_labels = self.view_projector.back_project_vertex_labels(
            mesh, np.stack(masks, axis=0)
        )

        return (mesh.cpu(), vertex_labels)
