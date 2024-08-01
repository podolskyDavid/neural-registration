import torch
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup


# 1. Load a pre-trained NeRF nerfacto model


class NeRFInverter:
    def __init__(self, path_to_config: str):
        _, self.pipeline, _, _ = eval_setup(config_path=Path(path_to_config), test_mode="inference")


if __name__ == "__main__":
    nerf_inverter = NeRFInverter(
        "/n/home10/dpodolskyi/neural-registration/outputs/0_065_cat5_2/nerfacto/2024-07-30_174955/config.yml")

    print(nerf_inverter.pipeline.device)

