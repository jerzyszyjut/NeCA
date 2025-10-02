# NeCA: 3D Coronary Artery Tree Reconstruction from Two 2D Projections via Neural Implicit Representation

# 1. Overview

This is the official code repository for the [NeCA](https://www.mdpi.com/2306-5354/11/12/1227) paper by Yiying Wang, Abhirup Banerjee and Vicente Grau, published at Bioengineering (Invited: free of APC).

## Citation

If you find the code useful, please consider citing the paper.

```
@Article{wang2024neca3dcoronaryartery,
AUTHOR = {Wang, Yiying and Banerjee, Abhirup and Grau, Vicente},
TITLE = {NeCA: 3D Coronary Artery Tree Reconstruction from Two 2D Projections via Neural Implicit Representation},
JOURNAL = {Bioengineering},
VOLUME = {11},
YEAR = {2024},
NUMBER = {12},
ARTICLE-NUMBER = {1227},
URL = {https://www.mdpi.com/2306-5354/11/12/1227},
ISSN = {2306-5354},
DOI = {10.3390/bioengineering11121227}
}
```

# 2. Introduction

Cardiovascular diseases (CVDs) are the most common health threats worldwide. 2D X-ray invasive coronary angiography (ICA) remains the most widely adopted imaging modality for CVD assessment during real-time cardiac interventions. However, it is often difficult for the cardiologists to interpret the 3D geometry of coronary vessels based on 2D planes. Moreover, due to the radiation limit, often only two angiographic projections are acquired, providing limited information of the vessel geometry and necessitating 3D coronary tree reconstruction based only on two ICA projections. In this paper, we propose a self-supervised deep learning method called NeCA, which is based on neural implicit representation using the multiresolution hash encoder and differentiable cone-beam forward projector layer, in order to achieve 3D coronary artery tree reconstruction from two 2D projections. We validate our method using six different metrics on a dataset generated from coronary computed tomography angiography of right coronary artery and left anterior descending artery. The evaluation results demonstrate that our NeCA method, without requiring 3D ground truth for supervision or large datasets for training, achieves promising performance in both vessel topology and branch-connectivity preservation compared to the supervised deep learning model.

## Our Proposed Model Architecture

<p align="center">
  <img src="https://github.com/WangStephen/NeCA/blob/main/img/model.svg">
</p>

# 3. Packages Requirement

For this repo to work use `uv` package manager to set up the environment:

```
uv sync
uv pip install setuptools==59.5.0
```

# 4. Code Instructions

## Data Preparation

Please prepare your projection data in shape `(1, Number of projections, Height, Weight)` and put the data under the `./data/CCTA_test/` folder. We recommend to use [ODL](https://github.com/odlgroup/odl) to generate your simulated projections from 3D coronary tree, which is tested and incorporated in this work.

Then please update the corresponding projection geometry in the file `./data/config.yml`.

## Model Optimisation

Configure your model hyper-parameters in the file `./config/CCTA.yaml`.

Then run the model to start 3D reconstruction optimisation:

```
python train.py --config ./config/CCTA.yaml
```

The 3D reconstruction results during iterations are saved under the folder `./logs/`.

# 5. License

Please see [license](https://github.com/WangStephen/NeCA/blob/main/LICENSE).

# 6. Acknowledgement

NeCA model architectures are revised based on [NAF](https://github.com/Ruyi-Zha/naf_cbct) and [NeRP](https://github.com/liyues/NeRP).

Multi-resolution hash encoder is based on [torch-ngp](https://github.com/ashawkey/torch-ngp).

Differentiable cone-beam forward projection layer is based on [ODL](https://github.com/odlgroup/odl) and we also use it to generate our simulated projections to model for optimisation.
