# CAS Practical Machine Learning - Project Thesis

Project thesis for [CAS - Practical Machine Learning](https://www.bfh.ch/de/weiterbildung/cas/practical-machine-learning/). 

## Resources

- [PyTorch](https://pytorch.org/)
    - [Start Locally](https://pytorch.org/get-started/locally/)
- [PyTorch3D](https://pytorch3d.org/)
    - [Installation](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

- https://medium.com/data-science/how-to-render-3d-files-using-pytorch3d-ef9de72483f8

## Dev Setup

- [Using uv with Jupyter from VSCode](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code)
- [Jupytext](https://jupytext.org/)
  - [Jupytext Sync Extension for VSCode](https://jupytext.org/integrations/vs-code/)
- [Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

### Setup after cloning the repository

1. Sync Python project dependencies using uv:

    ```
    uv sync
    ```

1. Rebuild the Jupyter notebooks by running the following command:

    ```
    jupytext --sync .\notebooks\*.py
    ```

### Upgrade dependencies

```
uv sync --upgrade
```

### Launch fiftyone from cmd line

```
fiftyone app launch Teeth2D
```

## Datasets / Challenges

- [Teeth3DS+: An Extended Benchmark for Intraoral 3D Scans Analysis](https://arxiv.org/pdf/2210.06094)
    - [3DTeethSeg Challenge MICCAI 2022](https://crns-smartvision.github.io/teeth3ds/#3DTeethSegSec)
        - [Dataset](https://osf.io/xctdy/)
    - [3DTeethLand Challenge MICCAI 2024](https://crns-smartvision.github.io/teeth3ds/#3DTeethLandSec)
        - [Dataset](https://osf.io/um96h/)

## Papers / Code

### Challenges
- [3DTeethSeg’22: 3D Teeth Scan Segmentation andLabeling Challenge](https://arxiv.org/abs/2305.18277)
- [Detecting Dental Landmarks from Intraoral 3D Scans: the 3DTeethLand challenge](https://arxiv.org/abs/2512.08323)

### Solutions
- [3DTeethSAM: Taming SAM2 for 3D Teeth Segmentation](https://arxiv.org/abs/2512.11557)
    - https://github.com/Crisitofy/3DTeethSAM
- [ToothInstanceNet: Comprehensive Information from Intra-oral Scans by Integration of Large-Context and High-Resolution Predictions](https://link.springer.com/chapter/10.1007/978-3-031-88977-6_21)
    - https://github.com/nnistelrooij/3dteethland
- [ToothGroupNetwork]()
    - https://github.com/limhoyeon/ToothGroupNetwork

## Issues & Workarounds

[Read more here](doc/issues.md)
