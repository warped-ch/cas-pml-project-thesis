# Notes on Experiments

## rf-detr/

[](../output/rf_detr_train/)

```

```

## rf-detr/20260815_213112

[20260815_213112](../output/rf_detr_train/20260815_213112)

- fix mesh backside rendering (still room for improvement)
- use composite feature images for Teeth2D dataset (grayscale, depth, curvature)

```
epochs=100,
batch_size=8,
grad_accum_steps=2,
lr=1e-4,
aug_config=AUG_CONSERVATIVE,
multi_scale=False,
use_ema=False,
pin_memory=True,
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
