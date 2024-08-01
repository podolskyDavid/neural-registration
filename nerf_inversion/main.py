import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.field_components.field_heads import FieldHeadNames


class NeRFInverter:
    def __init__(self, path_to_config: str):
        _, self.pipeline, _, _ = eval_setup(config_path=Path(path_to_config), test_mode="inference")
        self.model = self.pipeline.model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.device = self.pipeline.device

        # Freeze the NeRF model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize camera optimizer
        self.camera_optimizer = CameraOptimizer(
            config=self.model.config.camera_optimizer,
            num_cameras=1,
            device=self.device
        )

        # Set up optimizer for camera parameters
        self.optimizer = optim.Adam(self.camera_optimizer.parameters(), lr=1e-3)

        # Loss function
        self.loss_fn = nn.MSELoss()

    def create_ray_bundle(self, camera_params: torch.Tensor) -> RayBundle:
        # Unpack camera parameters
        position, rotation, fov = camera_params.split([3, 3, 1], dim=-1)

        # Create Cameras object
        cameras = Cameras(
            camera_to_worlds=torch.eye(4).unsqueeze(0).to(self.device),
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            width=self.pipeline.datamanager.train_dataset.cameras.width,
            height=self.pipeline.datamanager.train_dataset.cameras.height,
        )

        # Apply camera parameters
        cameras.position = position
        cameras.rotation = rotation
        cameras.fov = fov

        # Generate ray bundle
        ray_bundle = cameras.generate_rays(camera_indices=torch.zeros(1, device=self.device).long())

        return ray_bundle

    def generate_rgb(self, camera_params: torch.Tensor) -> torch.Tensor:
        # Create a RayBundle from camera parameters
        ray_bundle = self.create_ray_bundle(camera_params)

        # Apply camera optimizer
        self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # Get outputs from the model
        with torch.no_grad():
            outputs = self.model.get_outputs(ray_bundle)

        # Extract RGB values
        rgb = outputs[FieldHeadNames.RGB]

        return rgb

    def optimize_camera(self, target_image: torch.Tensor, num_iterations: int = 1000) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # Initialize camera parameters (position, rotation, fov)
        camera_params = torch.zeros(1, 7, device=self.device, requires_grad=True)

        for i in range(num_iterations):
            self.optimizer.zero_grad()

            # Generate RGB image from current camera parameters
            generated_rgb = self.generate_rgb(camera_params)

            # Compute loss
            loss = self.loss_fn(generated_rgb, target_image)

            # Backpropagate and optimize
            loss.backward()
            self.optimizer.step()

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")

        # Return optimized camera parameters and final generated image
        return camera_params.detach(), self.generate_rgb(camera_params).detach()


if __name__ == "__main__":
    print("Creating NeRF inverter...")
    nerf_inverter = NeRFInverter(
        "/n/home10/dpodolskyi/neural-registration/outputs/0_065_cat5_2/nerfacto/2024-07-30_174955/config.yml")
    print("NeRF inverter created.")
    print(f"Sanity check. Loading the cuda device: {nerf_inverter.pipeline.device}")

    gen_img = nerf_inverter.generate_rgb(torch.zeros(1, 7, requires_grad=True))
    print(f"Generated RGB image shape: {gen_img}")
    print(f"Shape of the generated RGB image: {gen_img.shape}")

    # Create a target image (replace this with your actual target image)
    # target_image = torch.rand(1, 512, 512, 3, device=nerf_inverter.device)
    #
    # # Optimize camera parameters
    # optimized_camera_params, final_image = nerf_inverter.optimize_camera(target_image)
    #
    # print("Optimized camera parameters:", optimized_camera_params)
    # print("Final image shape:", final_image.shape)
