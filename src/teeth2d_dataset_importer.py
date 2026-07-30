from pathlib import Path
from typing import Any

import cv2
import fiftyone as fo
import fiftyone.utils.data as foud


class Teeth2DDatasetImporter(foud.LabeledImageDatasetImporter):
    """
    Custom FiftyOne importer for "Teeth2D" dataset, to load images and multiclass segmentation masks.
    
    Note: "background hack"
    Pixel value 0 is reserved for "background" in fiftyone. 
    To overcome this, "class_0" is remapped to pixel value 1 when loading mask image. 
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

        self.mask_targets = {int(cid): f"class_{cid}" for cid in config["class_ids"]}
        # see note: "background hack"
        self.mask_targets[1] = self.mask_targets.pop(0)

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
        # see note: "background hack"
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 0] = 1

        # semantic segmentation labels
        segmentation = fo.Segmentation(mask=mask)

        # convert semantic segmentation labels to instance segmentation labels (fo.Detections):
        # - improved label visualization (selective display)
        # - COCO export
        detections = segmentation.to_detections(mask_targets=self.mask_targets)
        # set iscrowd attribute for detections, indicates the segment encompasses a group of objects (relevant for thing categories)
        for det in detections.detections:
            det["iscrowd"] = 1

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
        return {"default_mask_targets": self.mask_targets}

    @property
    def label_cls(self):
        # return a dictionary type mapping since we output multiple fields
        return {"ground_truth_seg": fo.Segmentation, "ground_truth_det": fo.Detections}

    def __len__(self):
        return len(self._uuids)
