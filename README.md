# NeRF-based Intraoperative Registration

## Overview
This repository implements advanced intraoperative brain registration using Neural Radiance Fields (NeRFs) as learned functional representations of brain surfaces. Unlike traditional mesh-based approaches, NeRFs provide a differentiable, implicit function of the brain's geometry and appearance, enabling optimization of camera positions through backpropagation for precise alignment of preoperative and intraoperative brain images.

## In my own words

**At its basics, the idea of the algorithm is the following:**

### What we have:
- A pre-trained NeRF from brain MR scans (density and coloring). This NeRF is an “implicit, differentiable representation” of the brain surface. Meaning we can backprop through it.
- During the surgery we can take a snapshots of the brain surface.
### **Goal:**
- Iteratively, through backpropagation of an error, align the camera angle, so that the final iterated snapshot of the NeRF is equal (up to a certain error) to the goal image (real brain surface image).
### Assumptions and the setup:
- Pre-trained NeRF’s density and coloring is a “perfect” representation of the real brain surface (perfecting the coloring and the density of the NeRF is a separate problem). Therefore, we simulate the experiment by sampling both the iterative snapshots and the final goal image form the same NeRF model (to eliminate lighting and other relevant noise. this is the next step in the research)
- The distance from the object is constant. E.g., whenever we take a snapshot it is always going to be from the same distance as the final goal image
### Steps:
1. Choose a “good” starting point in 3d nerf space. Basically, we are choosing a transformation matrix (angles + x/y/z).
        - A good angle is the one that captures the targeted brain surface area, instead of pointing in a void. (next step in improving the algorithm would be automatically choosing a good angle, e.g., even when we don’t see the brain surface. but this is more of a software engineering problem and outside of this work)
2. Take a snapshot of the brain (again, sampled from the same NeRF)
3. Calculate the error between the goal snapshot of the brain surface and the current iteration snapshot (depends on the error that we are exploring)
4. Backpropagate through the error and adapt the transformation matrix
5. Repeat points two through 5 until error < *e*

## Research Focus
This work extends the methodology presented in "Intraoperative Registration by Cross-Modal Inverse Neural Rendering" with:

1. **Loss Function Exploration**: Implementation and analysis of multiple loss functions:
   - Mutual Information
   - Normalized Cross-Correlation
   - Weighted/masked L2 loss

2. **Hypernetwork Style Transfer**: Analysis of hypernetwork-generated styles on registration accuracy, with various image encoding methods:
   - Y'UV color space
   - Histogram of Oriented Gradients (HOG)
   - Texture-based features (e.g., Gabor filters)
   - Edge detection & contour matching
   - Gram matrices for style consistency
   - Deep feature matching via pre-trained CNNs
   - Structural Similarity Index (SSIM)

## Technical Implementation
- Built on top of nerfstudio to leverage its implementation abstractions
- NeRF-implementation agnostic (works with Vanilla NeRF, Instant-NGP, Nerfacto, etc.)
- Focused on 6 Degrees of Freedom (6DoF) estimation during surgery
- Improves upon previous implementations:
  - Unlike iNeRF: NeRF- and loss-implementation agnostic
  - Extends Parallel Inversion: Implements additional loss functions beyond L2

## Project Roadmap

### Current Development
- [ ] Finalize nerfstudio abstraction implementation
- [ ] Complete loss module implementations
- [ ] Implement hypernetwork modules
- [ ] Experiment with different hypernetwork methods and loss functions

### Future Work
- [ ] Explore additional hypernetwork architectures
- [ ] Enhance registration via supplementary MLP modules
- [ ] Investigate alternative implicit representation methods for brain surfaces

## Applications
This research aims to improve the precision of image-guided neurosurgery by enhancing the accuracy of intraoperative brain registration, ultimately leading to better surgical outcomes.

