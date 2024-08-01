from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel  # for subclassing InstantNGP model
from style_ngp_field import StyleNGPField

# import all the other necessary imports
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import VolumetricSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.model_components.losses import MSELoss

from torch.nn import Parameter
import nerfacc

import torch
import torch.optim as optim


@dataclass
class StyleNGPModelConfig(InstantNGPModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: StyleNGPModel)


class StyleNGPModel(NGPModel):
    """Template Model."""

    config: StyleNGPModelConfig
    field: StyleNGPField

    def populate_modules(self):
        # Calling superclass
        super().populate_modules()

        # Rest adapted from instant-ngp
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = StyleNGPField(
            aabb=self.scene_box.aabb,
            # TODO: is this correct? Why 0 if use?
            num_images=self.num_train_data,
            log2_hashmap_size=self.config.log2_hashmap_size,
            max_res=self.config.max_res,
        )

        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.render_step_size is None:
            # auto step size: ~1000 samples in the base level grid
            self.config.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / 1000
        # Occupancy Grid.
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.config.grid_resolution,
            levels=self.config.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    def optimize_xyz(self, target_position, initial_position, angles, lr=0.01, num_iterations=1000):
        # Convert initial position to tensor
        position = torch.tensor(initial_position, requires_grad=True, device=self.device)

        # Target image
        target_image = self.generate_image(target_position, angles)
        target_image = torch.tensor(target_image, requires_grad=False, device=self.device)

        # Optimizer
        optimizer = optim.Adam([position], lr=lr)

        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # Generate the starting image with the current position
            starting_image = self.generate_image(position, angles)
            starting_image = torch.tensor(starting_image, requires_grad=False, device=self.device)

            # Compute the loss (MSE)
            loss = torch.mean((starting_image - target_image) ** 2)

            # Backpropagation
            loss.backward()

            # Update the position
            optimizer.step()

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")

        optimized_position = position.detach().cpu().numpy()
        print(f"Optimized Position: {optimized_position}")
        return optimized_position

    def generate_image(self, position, angles):
        # Generate the image using the position and angles
        ray_bundle = self.get_ray_bundle(position, angles)
        outputs = self.get_outputs(ray_bundle)
        image = outputs['rgb'].cpu().detach().numpy()
        return image

    def get_ray_bundle(self, position, angles):
        # Create a RayBundle from position and angles
        # Implement this based on how your model expects the ray inputs
        pass
