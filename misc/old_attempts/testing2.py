import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from pathlib import Path

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.trainer import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.eval_utils import eval_setup


class NerfInverter(nn.Module):
    """
    NerfInverter class that uses a pre-trained NerfactoModel and optimizes camera parameters.

    This class encapsulates the pipeline for optimizing camera parameters for a pre-trained NeRF model.

    Attributes:
        model (NerfactoModel): The pre-trained NerfactoModel.
        camera_optimizer (CameraOptimizer): Optimizer for camera parameters.
    """

    def __init__(self, path_to_config: str, camera_optimizer_config: CameraOptimizerConfig):
        super().__init__()

        # Load the pre-trained NerfactoModel
        _, self.pipeline, _, _ = eval_setup(config_path=Path(path_to_config), test_mode="inference")
        self.model: NerfactoModel = self.pipeline.model

        # Freeze the NeRF model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize the camera optimizer
        self.camera_optimizer = camera_optimizer_config.setup(
            num_cameras=self.model.num_train_data,
            device=self.model.device
        )

    def forward(self, ray_bundle: RayBundle):
        """
        Forward pass of the NerfInverter.

        Args:
            ray_bundle (RayBundle): The input ray bundle.

        Returns:
            dict: The output of the NerfactoModel's get_outputs method.
        """
        # Apply camera optimization
        self.camera_optimizer.apply_to_raybundle(ray_bundle)

        # Get outputs from the pre-trained model
        outputs = self.model.get_outputs(ray_bundle)

        return outputs

    def get_metrics_dict(self, outputs, batch):
        """
        Compute the metrics dictionary.

        Args:
            outputs (dict): The output from the forward pass.
            batch (dict): The input batch data.

        Returns:
            dict: The computed metrics.
        """
        metrics_dict = self.model.get_metrics_dict(outputs, batch)
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """
        Compute the loss dictionary.

        Args:
            outputs (dict): The output from the forward pass.
            batch (dict): The input batch data.
            metrics_dict (dict, optional): The metrics dictionary.

        Returns:
            dict: The computed losses.
        """
        loss_dict = self.model.get_loss_dict(outputs, batch, metrics_dict)
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Compute image metrics and get output images.

        Args:
            outputs (dict): The output from the forward pass.
            batch (dict): The input batch data.

        Returns:
            tuple: A tuple containing the image metrics and output images.
        """
        return self.model.get_image_metrics_and_images(outputs, batch)

    def get_param_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get the parameter groups for optimization.

        Returns:
            dict: A dictionary of parameter groups.
        """
        param_groups = {}
        self.camera_optimizer.get_param_groups(param_groups)
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """
        Get the training callbacks.

        Args:
            training_callback_attributes (TrainingCallbackAttributes): Attributes for training callbacks.

        Returns:
            list: A list of training callbacks.
        """
        return self.model.get_training_callbacks(training_callback_attributes)