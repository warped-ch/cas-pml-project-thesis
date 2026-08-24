# Notes on Experiments


## rf-detr/

[](../output/rf_detr_train/)

```py

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
# TODO: disable for final training run
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
# TODO: disable for final training run
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
