# Enhancing Intraoperative Registration with Neural Radiance Fields

## Research Overview

This repository contains the implementation and findings from my bachelor's thesis: "Enhancing Intraoperative Registration with Neural Radiance Fields: An Exploration of Loss Functions Effects." The project advances intraoperative brain registration by leveraging Neural Radiance Fields (NeRFs) as differentiable, implicit representations of brain surfaces, with a specific focus on how various loss functions impact registration accuracy and convergence characteristics.

## Background & Motivation

Precise alignment of preoperative and intraoperative medical images is fundamental for successful image-guided neurosurgery. Traditional registration methods typically rely on mesh-based approaches, which present inherent limitations in terms of flexibility and differentiability. This work explores the use of Neural Radiance Fields (NeRFs) as an alternative representation that offers several key advantages:

- NeRFs provide a continuous, differentiable function representing both the geometry and appearance of the brain surface
- Unlike discrete mesh representations, NeRFs enable direct optimization through backpropagation
- The implicit nature of NeRFs allows for high-fidelity representation of complex anatomical structures

This research builds upon the foundation laid by "Intraoperative Registration by Cross-Modal Inverse Neural Rendering" by Maximilian Fehrentz et al., which introduced using neural implicit representations for surgical registration with hypernetworks for appearance modeling and parallel inversion for pose estimation. Our work extends this approach with a more customizable implementation and a focused analysis of loss function impacts.

## Technical Approach

The algorithm operates on the following principle:

1. **Input**: A pre-trained NeRF model of the brain derived from MR scans, representing both density and color information
2. **Target**: Intraoperative snapshots of the exposed brain surface during surgery
3. **Objective**: Iteratively optimize camera parameters through backpropagation to align rendered NeRF views with the target intraoperative image

The core process involves:

1. Selecting an initial camera position and orientation (transformation matrix)
2. Rendering a view from the NeRF at the current camera parameters
3. Computing the error/dissimilarity between the rendered image and the target image using a specific loss function
4. Backpropagating this error to update the camera parameters
5. Repeating until convergence or maximum iterations

The key innovation in our implementation is the use of finite differences to calculate gradients, enabling stable optimization across different loss functions while maintaining compatibility with various NeRF architectures within the nerfstudio framework.

## Implementation Details

The `iNeRFOptimizerBatchedFD` class implements our approach with several key features:

- **Model Agnosticism**: Works with any NeRF model implementation in nerfstudio (vanilla, instantNGP, etc.)
- **Customizable Loss Functions**: Supports various loss functions for comparing rendered and target images
- **Batched Finite Differences**: Efficiently computes gradients for camera parameter optimization
- **Comprehensive Tracking**: Records detailed experiment data including loss trajectories and visualizations
- **Robust Error Handling**: Implements fallback mechanisms to ensure experiment data is preserved

The implementation addresses previous limitations by:
- Enabling customization of loss functions (not limited to L2 loss)
- Maintaining compatibility with nerfstudio workflows for easier integration with hypernetwork-colored models
- Eliminating the need to retrain NeRF models after coloring with hypernetworks

## Loss Functions Explored

We implemented and analyzed five different loss functions:

1. **L1 Loss (Mean Absolute Error)**
   - Measures the absolute difference between pixels
   - Less sensitive to outliers compared to L2
   - Promotes sparsity in the error distribution

2. **L2 Loss (Mean Squared Error)**
   - Standard approach measuring squared differences between pixels
   - More heavily penalizes large deviations

3. **Structural Similarity Index Loss (SSIM)**
   - Considers structural information in images
   - Evaluates patterns of pixel intensities based on local luminance and contrast
   - More perceptually aligned than pixel-wise metrics

4. **Normalized Cross-Correlation Loss (NCC)**
   - Measures the correlation between images normalized by their standard deviations
   - Robust to linear intensity variations
   - Particularly useful for cross-modal registration

5. **Mutual Information Loss (MI)**
   - Information-theoretic measure of statistical dependency
   - Highly robust to intensity variations and across imaging modalities
   - Implemented with soft binning for differentiability

## Experimental Setup & Methodology

Our experiments were conducted with the following controlled parameters:

- **Iteration Limit**: 50 iterations per experiment
- **Optimizer**: AdamW with learning rate of 0.01
- **Test Matrix**: 10 different starting points paired with the same target image
- **Rendering Source**: Both target images and starting renders were sourced from the same NeRF to control for representation quality
- **Performance Metrics**: Convergence speed, final alignment quality, and optimization path characteristics

We deliberately constrained the experimental space to isolate the effect of different loss functions on the registration process, maintaining all other variables constant.

## Results & Analysis

Our findings revealed distinct characteristics for each loss function:

### L1 Loss
- **Convergence Speed**: Fastest average convergence (best loss by iteration 39)
- **Optimization Path**: Smooth, direct trajectories with minimal deviations
- **Time Performance**: Average 621 seconds for 50 iterations
- **Qualitative Assessment**: Excellent alignment with high precision

### L2 Loss
- **Convergence Speed**: Good convergence by iteration 47
- **Optimization Path**: Smooth trajectories similar to L1
- **Time Performance**: Average 624 seconds for 50 iterations
- **Qualitative Assessment**: Good alignment but slightly slower than L1

### Structural Similarity Index Loss
- **Convergence Speed**: Slower convergence, reaching best loss by iteration 49
- **Optimization Path**: More exploratory in initial iterations
- **Time Performance**: Average 626 seconds for 50 iterations
- **Qualitative Assessment**: Good perceptual alignment but takes longer to stabilize

### Normalized Cross-Correlation Loss
- **Convergence Speed**: Moderate convergence by iteration 44
- **Optimization Path**: "Jiggly" with more exploration
- **Time Performance**: Average 621 seconds for 50 iterations
- **Qualitative Assessment**: More robust to NeRF artifacts but less consistent trajectories

### Mutual Information Loss
- **Convergence Speed**: Good convergence by iteration 42
- **Optimization Path**: Initially exploratory, then stabilizing
- **Time Performance**: Average 621 seconds for 50 iterations
- **Qualitative Assessment**: Most robust to artifacts but with irregular optimization paths

### Key Takeaways
- L1 loss consistently outperformed other measures in terms of convergence speed
- Information-theoretic and correlation-based measures (MI, NCC) demonstrated greater robustness to NeRF rendering artifacts
- All loss functions eventually converged to acceptable solutions, suggesting the choice depends on specific use-case priorities (speed vs. robustness)
- The more complex loss functions (MI, SSIM, NCC) exhibited more exploratory behavior in early iterations

## Limitations & Challenges

Our approach faces several limitations that warrant consideration:

1. **Gradient Computation**: Use of finite differences instead of direct backpropagation introduces approximation errors and computational overhead

2. **Representation Fidelity**: We assume the pre-trained NeRF perfectly represents the brain surface, which is a significant simplification. In reality, factors such as lighting variations, specular reflections, and tissue deformation create discrepancies.

3. **Computational Efficiency**: The current implementation is too slow for real-time surgical applications, with iteration times of approximately 12-13 seconds.

4. **Sample Size**: The experiments were conducted on a limited dataset, potentially affecting the generalizability of our findings.

5. **Technical Challenges**: The original implementation encountered issues with gradient flow disconnection in the computational graph, necessitating the finite differences approach.

## Future Directions

Based on our findings and identified limitations, we propose several avenues for future research:

1. **Alternative Neural Representations**: Exploration of Gaussian Splats as a faster, higher-quality alternative to NeRFs. These representations offer potentially better rendering quality with significantly reduced computational requirements.

2. **Gradient Flow Optimization**: Returning to proper backpropagation by resolving the technical issues with gradient disconnection.

3. **Appearance Modeling**: Investigating the integration of hypernetworks for improved coloring of brain surfaces to better match intraoperative appearances.

4. **Broader Evaluation**: Conducting more extensive experimentation with larger datasets, varied anatomical regions, and different parameter settings.

5. **Starting Point Selection**: Developing techniques for automated selection of good initial camera positions to improve convergence reliability.

6. **Multi-Loss Approaches**: Exploring combined or adaptive loss functions that leverage the strengths of different metrics throughout the optimization process.

## Significance & Applications

This research has several potential impacts on image-guided neurosurgery:

- **Improved Registration Accuracy**: The differentiable nature of NeRFs enables more precise alignment between preoperative and intraoperative images.

- **Real-time Adaptation**: With further optimization, this approach could potentially adapt to tissue deformation and movement during surgery.

- **Reduced Invasiveness**: Better registration may reduce the need for invasive fiducial markers currently used in neurosurgical navigation.

- **Cross-modal Registration**: The exploration of various loss functions lays groundwork for better registration across different imaging modalities.

The presented approach, while still in research stages, demonstrates the potential of neural implicit representations to advance the state-of-the-art in surgical navigation and planning.