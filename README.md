# Intraoperative Registration via Cross-Modal Inverse Neural Rendering
This repository is the accompanying code to the paper "Intraoperative Registration via Cross-Modal Inverse Neural Rendering". It contains the code for the multistyle NeRF and the pose estimation method. Our NeRF is based on the Nerfstudio implementation of Instant-NGP and the actual registration is done using Parallel Inversion. Since there is no pytorch implementation of a SOTA solver, we use the NeRF to infer the target style and retrain the model in the Parallel Inversion environment to register that NeRF to the intraoperative image.

## Requirements

- NVIDIA GPU

## Installation

To configure the environment to work with Nerfstudio, Parallel Inversion, and other dependencies, follow the instructions below.
```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118 conda env create -f environment.yaml
conda activate style-ngp
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

> I had to install pymeshlab==2023.12.post1 from the source
```bash
wget https://github.com/cnr-isti-vclab/PyMeshLab/archive/refs/tags/v2023.12.post1.tar.gz
tar -xvzf v2023.12.post1.tar.gz
cd PyMeshLab-2023.12.post1
pip install .
```

Register the method in Nerfstudio to later run our method in the Nerfstudio environment.
```bash
pip install -e .
ns-install-cli
```

## Getting Started