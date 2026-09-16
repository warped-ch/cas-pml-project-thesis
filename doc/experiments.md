# Notes on Experiments

## rf-detr/

[](../output/rf_detr_train/)

```py

```

## rf-detr/20260915_185454

[20260915_185454](../output/rf_detr_train/20260915_185454)

- next best custom dataset (greedy train split, still respecting official test split)

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=1,
lr=1e-4,
aug_config={},  # disable augmentation (no horizontal flip)
save_dataset_grids=True,
multi_scale=False,
# eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
compute_val_loss=True,
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260912_010327

[20260912_010327](../output/rf_detr_train/20260912_010327)

- lowered grad_accum_steps from 2 to 1
- eval_interval to default: 1
- upgraded dependencies (rfdetr from 1.9.4 to 1.10.1, ...)

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=1,
lr=1e-4,
aug_config={},  # disable augmentation (no horizontal flip)
save_dataset_grids=True,
# TODO: speed up training by specifying "num_queries" according to classes in dataset?
multi_scale=False,
#eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
compute_val_loss=True,
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260909_191810

[20260909_191810](../output/rf_detr_train/20260909_191810)

- disable augmentation
- consider official train/test split
- train on the following custom datasets
  - Teeth2D_lower
  - Teeth2D_upper
  - Teeth2D

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
aug_config={},  # disable augmentation (no horizontal flip)
save_dataset_grids=True,
# TODO: speed up training by specifying "num_queries" according to classes in dataset?
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260909_035324

[20260909_035324](../output/rf_detr_train/20260909_035324)

- ⚠️ default augmentation with horizontal flip!
- consider official train/test split creation of custom train/valid/test splits
- train on specialized custom datasets
  - Teeth2D_lower
  - Teeth2D_upper

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
save_dataset_grids=True,
# TODO: speed up training by specifying "num_queries" according to classes in dataset?
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260829_114546

[20260829_114546](../output/rf_detr_train/20260829_114546)

- add light rotation for augmentation

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
aug_config={
    "transforms": [
        {"type": "Rotate", "limit": 10, "p": 0.3, "border_mode": 0}
    ]
},
save_dataset_grids=True,
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260828_064008

[20260828_064008](../output/rf_detr_train/20260828_064008)

- train on specialized custom datasets
  - Teeth2D_lower
  - Teeth2D_upper
- train on full dataset Teeth2D
- set learning rate back to default
- enable save_dataset_grids for visual sanity checks

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
aug_config={},  # disable horizontal flip while keeping required resizing and normalization
save_dataset_grids=True,
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260826_212828

[20260826_212828](../output/rf_detr_train/20260826_212828)

- incomplete training run for testing, train on Teeth2D_lower only
- reduce learning rate from 1e-4 to 5e-5 again

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=5e-5,
aug_config={},  # disable horizontal flip while keeping required resizing and normalization
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260826_200249

[20260826_200249](../output/rf_detr_train/20260826_200249)

- incomplete training run for testing, train on Teeth2D_lower only
- keep context awareness for view projection (see comments on `views` in [config.yaml](../config/config.yaml))
- disable flip for augmentation (see [Notes on augmentation](../notebooks/40_train_rfdetr.ipynb) in training notebook)

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
aug_config={},  # disable horizontal flip while keeping required resizing and normalization
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260825_211334

[20260825_211334](../output/rf_detr_train/20260825_211334)

- incomplete training run for testing, train on Teeth2D_lower only
- upgrade rf-detr dependency to "rfdetr[augment,loggers,train]>=1.9.1"
  - significantly reduced CPU load (~10% vs ~40%)
- temp hack to get the same result from inference pipeline as when loading image from file
- bring back majority voting for 3D back projection
- fix "flat gray" mesh texture problem by enabling backface culling

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=5e-5,
aug_config=AUG_CONSERVATIVE,
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## ⚠️ training runs below had augmentation bug (horizontal flip)

## rf-detr/20260825_202730

[20260825_202730](../output/rf_detr_train/20260825_202730)

- incomplete training run on Teeth2D_lower only, just to see how we're doing...

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=5e-5,
aug_config=AUG_CONSERVATIVE,
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260823_145349

[20260823_145349](../output/rf_detr_train/20260823_145349)

Only trained 2 models on a subset of the data for testing (Teeth2D_lower_has_model_base_false, Teeth2D_upper_has_model_base_false).

- optimize view projection parameters
  - rotate front center and back center views so that teeth labels have same orientation across all views
  - increase tilt on side views to see more teeth

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=5e-5,
aug_config=AUG_CONSERVATIVE,
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
use_ema=False,
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260822_201212

[20260822_201212](../output/rf_detr_train/20260822_201212)

- custom train/valid/test splits (focus on missing_teeth, has_model_base)

```py
dataset_dir=str(dataset_path),
output_dir=output_dir,
epochs=200,
batch_size=8,
grad_accum_steps=2,
lr=5e-5,
aug_config=AUG_CONSERVATIVE,
multi_scale=False,
eval_interval=5,
early_stopping=True,
early_stopping_patience=10,  # Wait 10 epochs before stopping
early_stopping_min_delta=0.005,  # Require 0.5% validation metric improvement
use_ema=False,
pin_memory=True,
progress_bar="tqdm",
```

## rf-detr/20260814_201756

[20260814_201756](../output/rf_detr_train/20260814_201756)

```py
epochs=100,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
aug_config=AUG_CONSERVATIVE,
multi_scale=False,
use_ema=False,
pin_memory=True,
```

## rf-detr/20260813_213740

[20260813_213740](../output/rf_detr_train/20260813_213740)

- ignore potentially private test set (to be checked), create random train/test/val splits based on Teeth3DS+ train split only (80%, 10%, 10%)
- use FoVOrthographicCameras instead of FoVPerspectiveCameras for multi-view projection rendering
- improve lights for multi-view projection rendering (widen histogram)
- consider config background_value for multi-view projection images (black background for view images)

```
epochs=100,
batch_size=4,
grad_accum_steps=8,
lr=2.5e-5,
use_ema=False,
```

## rf-detr/20260810_230858

- [20260810_230858](../output/rf_detr_train/20260810_230858)

```
epochs=100,
batch_size=4,
grad_accum_steps=8,
lr=2.5e-5,
```

## rf-detr/20260809_164735

- [20260809_164735](../output/rf_detr_train/20260809_164735)

```
epochs=50,
batch_size=4,
grad_accum_steps=16,
lr=5e-5,
```

## rf-detr/20260808_205728

- [20260808_205728](../output/rf_detr_train/20260808_205728)

```
epochs=25,
batch_size=4,
grad_accum_steps=16,
lr=5e-5,
```
