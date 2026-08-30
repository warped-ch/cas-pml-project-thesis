from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
import fiftyone.utils.data as foud


# TODO: possible to speed up fo stuff using multi-precessing, batch processing?
class Teeth2DDatasetImporter(foud.LabeledImageDatasetImporter):
    """
    Custom FiftyOne importer for "Teeth2D" dataset, to load images and multiclass segmentation masks.
    
        - https://docs.voxel51.com/user_guide/using_datasets.html#semantic-segmentation
        - https://docs.voxel51.com/user_guide/using_datasets.html#instance-segmentations

    """

    def __init__(
        self,
        config: dict[str, Any],
        dataset_dir,
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        super().__init__(
            dataset_dir, shuffle=shuffle, seed=seed, max_samples=max_samples
        )

        self.config = config

        self.default_mask_targets = {}
        class_id_map = self.config["class_id_map"]
        for cid in self.config["class_ids"]:
            old_id = int(cid)
            new_id = class_id_map.get(old_id, old_id) if class_id_map else old_id
            # keep the old class id encoded in class name, will be used to restore original class id when generating final predicted mask
            self.default_mask_targets[new_id] = f"class_{old_id}"
        print(f"mask_targets={self.default_mask_targets}")

        self.default_classes = [
            self.default_mask_targets[cid] for cid in sorted(self.default_mask_targets.keys())
        ]

        self._dataset_root = Path(dataset_dir)
        self._images_dir = self._dataset_root / "images"
        self._masks_dir = self._dataset_root / "masks"

        self._uuids = []
        self._iter_uuids = None

    def setup(self):
        # Find all images and matching masks by filename
        image_paths = list(self._images_dir.glob("*.png"))
        for image_path in image_paths:
            mask_path = self._masks_dir / image_path.name
            if mask_path.exists():
                self._uuids.append((str(image_path), str(mask_path)))

        self._uuids = self._preprocess_list(self._uuids)

    def __iter__(self):
        self._iter_uuids = iter(self._uuids)
        return self

    def __next__(self):
        image_path, mask_path = next(self._iter_uuids)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # semantic segmentation labels
        segmentation = fo.Segmentation(mask=mask)

        # convert semantic segmentation labels to instance segmentation labels (fo.Detections):
        # - improved label visualization (selective display)
        # - COCO export
        detections = segmentation.to_detections(mask_targets=self.default_mask_targets)

        label_dict = {
            "ground_truth_seg": segmentation,
            "ground_truth_det": detections,
        }

        # TODO: add metadata, tags?

        # return a 3-tuple structure: (image_path, image_metadata, labels)
        # pass None for image_metadata so FiftyOne computes it automatically
        return image_path, None, label_dict

    @property
    def has_dataset_info(self):
        return True

    def get_dataset_info(self):
        return {"default_mask_targets": self.default_mask_targets}

    @property
    def label_cls(self):
        # return a dictionary type mapping since we output multiple fields
        return {"ground_truth_seg": fo.Segmentation, "ground_truth_det": fo.Detections}

    def __len__(self):
        return len(self._uuids)
