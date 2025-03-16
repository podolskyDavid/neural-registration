import torch
import numpy as np
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle

class BrainSurfaceAligner:
    def __init__(
        self,
        pipeline,
        target_image,
        initial_transform=None,
        learning_rate=0.01,
        image_height=512,
        image_width=512,
        max_iterations=1000,
        convergence_threshold=1e-5,
    ):
        self.pipeline = pipeline
        self.model = pipeline.model
        self.device = self.model.device
        self.target_image = target_image.to(self.device)
        self.image_height = image_height
        self.image_width = image_width
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Freeze all model parameters to ensure we only optimize the camera
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Initialize camera parameters from provided transform or with defaults
        if initial_transform is None:
            # Default transformation looking along -z axis
            initial_transform = torch.eye(4, dtype=torch.float32)
        
        # Extract rotation and translation as separate parameters for optimization
        self.camera_rotation = torch.nn.Parameter(
            initial_transform[:3, :3].clone().to(self.device)
        )
        self.camera_translation = torch.nn.Parameter(
            initial_transform[:3, 3].clone().to(self.device)
        )
        
        # Create optimizer for camera parameters only
        self.optimizer = torch.optim.Adam(
            [self.camera_rotation, self.camera_translation], 
            lr=learning_rate
        )
        
        # Loss function
        self.loss_fn = torch.nn.MSELoss()
        
    def get_camera_to_world(self):
        """Create transformation matrix from current parameters"""
        c2w = torch.eye(4, dtype=torch.float32, device=self.device)
        c2w[:3, :3] = self.camera_rotation
        c2w[:3, 3] = self.camera_translation
        return c2w
        
    def create_camera(self):
        """Create nerfstudio Camera object from current parameters"""
        c2w = self.get_camera_to_world().unsqueeze(0)  # Add batch dimension
        
        # Use camera parameters from the first camera in training dataset 
        # or default parameters if not available
        try:
            fx = self.pipeline.datamanager.train_dataset.cameras.fx[0]
            fy = self.pipeline.datamanager.train_dataset.cameras.fy[0]
            cx = self.pipeline.datamanager.train_dataset.cameras.cx[0]
            cy = self.pipeline.datamanager.train_dataset.cameras.cy[0]
        except (AttributeError, IndexError):
            # Default camera intrinsics if not available
            fx = torch.tensor([1000.0], device=self.device)
            fy = torch.tensor([1000.0], device=self.device)
            cx = torch.tensor([self.image_width/2], device=self.device)
            cy = torch.tensor([self.image_height/2], device=self.device)
            
        camera = Cameras(
            camera_to_worlds=c2w,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=torch.tensor([self.image_width], dtype=torch.int32, device=self.device),
            height=torch.tensor([self.image_height], dtype=torch.int32, device=self.device),
        )
        return camera
        
    def render_view(self):
        """Render an image using the current camera parameters"""
        camera = self.create_camera()
        
        # Generate rays
        ray_bundle = camera.generate_rays(camera_indices=0)
        
        # Move to correct device
        ray_bundle = ray_bundle.to(self.device)
        
        # Set model to eval mode while maintaining gradients
        self.model.eval()
        
        # Critical: We need to maintain gradients during rendering
        with torch.set_grad_enabled(True):
            # Apply collider if present (same as model.forward)
            if self.model.collider is not None:
                ray_bundle = self.model.collider(ray_bundle)
                
            # Get outputs directly from model.get_outputs to keep gradients
            outputs = self.model.get_outputs(ray_bundle)
            
        return outputs["rgb"]
        
    def optimize_step(self):
        """Perform one optimization step"""
        self.optimizer.zero_grad()
        
        # Render image with current camera parameters
        rendered_image = self.render_view()
        
        # Compute loss against target image
        loss = self.loss_fn(rendered_image, self.target_image)
        
        # Backpropagate
        loss.backward()
        
        # Update camera parameters
        self.optimizer.step()
        
        # Orthogonalize rotation matrix to ensure it remains a valid rotation
        with torch.no_grad():
            u, _, v = torch.svd(self.camera_rotation)
            self.camera_rotation.data = u @ v.T
            
        return loss.item(), rendered_image
        
    def run_optimization(self, callback=None):
        """Run the full optimization process"""
        losses = []
        images = []
        
        for i in range(self.max_iterations):
            loss, rendered_image = self.optimize_step()
            losses.append(loss)
            
            if callback and i % 10 == 0:
                callback(i, loss, rendered_image, self.get_camera_to_world())
                
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")
                images.append(rendered_image.detach().cpu().numpy())
                
            # Check for convergence
            if i > 20 and abs(losses[-1] - losses[-20]) < self.convergence_threshold:
                print(f"Converged at iteration {i}")
                break
                
        return {
            "final_loss": losses[-1],
            "loss_history": losses,
            "image_history": images,
            "final_transform": self.get_camera_to_world().detach().cpu().numpy(),
            "final_rotation": self.camera_rotation.detach().cpu().numpy(),
            "final_translation": self.camera_translation.detach().cpu().numpy(),
        }