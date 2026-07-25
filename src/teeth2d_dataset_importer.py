from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
import fiftyone.utils.data as foud
import numpy as np


class Teeth2DDatasetImporter(foud.LabeledImageDatasetImporter):
    """Custom FiftyOne importer for "Teeth2D" dataset, to load images and multiclass segmentation masks."""

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
        self.background_value = config["2d_projection"]["background_value"]

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

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_array = np.array(mask_img)

        # find the unique class IDs (ignore background)
        class_ids = np.unique(mask_array)
        class_ids = class_ids[class_ids != self.background_value]

        # separate each class into its own binary mask layer stored in a dictionary
        label_dict = {}
        for class_id in class_ids:
            binary_mask = (mask_array == class_id).astype(np.uint8)
            label_dict[f"class_{class_id}"] = fo.Segmentation(mask=binary_mask)

        # return a 3-tuple structure: (image_path, image_metadata, labels)
        # pass None for image_metadata so FiftyOne computes it automatically
        return image_path, None, label_dict

    @property
    def has_dataset_info(self):
        return False

    @property
    def label_cls(self):
        # Return a dictionary type mapping since we output multiple fields
        return dict

    def __len__(self):
        return len(self._uuids)
