# RAY TRACER 

## Tech Stack Analysis
### 1. Image Generator (`kaggle_1.py`) - SEM Wafer Dataset Generator

#### Core Libraries
- **PyTorch** (`torch`) - GPU/CUDA support for tensor operations
  - Device management (CUDA/CPU detection)
  - Tensor operations for efficient image processing
  
- **NumPy** (`numpy`) - Numerical computing
  - Array operations for image synthesis
  - Mathematical computations (meshgrid, trigonometry, etc.)
  - Random number generation with seeded RNG

- **Pillow (PIL)** - Image processing and manipulation
  - Image loading/saving (PNG format)
  - Image drawing and filtering
  - Image evaluation and transformation

### Additional Utilities
- **csv** - Data manifest generation and storage
- **zipfile** - Dataset compression and packaging
- **time** - Performance timing and monitoring
- **os** - File system operations and path management

### Specialized Features
- **Advanced Image Synthesis**
  - Heterogeneous SEM (Scanning Electron Microscopy) tile generation
  - Multiple die architecture patterns:
    - Standard rectangular layout
    - Dense rectangular layout
    - Wide rectangular layout
    - Fine mesh crosshatch pattern
  
- **Procedural Generation**
  - Comet-dash via patterns with streaking effects
  - Deterministic seeded random generation for reproducibility
  - Gaussian blur-based line rendering
  - Sparse junction masking via hash-based selection
  
- **Geometric Transformations**
  - 2D rotation support (wafer stage rotation simulation)
  - Zoom/reference cropping (10x zoom for reference images)
  - Bounding box calculation and tracking
  
- **Dataset Output**
  - CSV manifest with metadata (paths, ground-truth coordinates, rotation, bbox)
  - Support for marked/unmarked search images
  - Compressed ZIP archive generation

---

## 2. Deep Learning Model (`ray-tracer-v3.ipynb`) - Wafer Reference Localization

### Core ML/DL Framework
- **PyTorch** (`torch`, `torch.nn`, `torch.nn.functional`)
  - Neural network architecture definition
  - Loss functions and optimization
  - GPU acceleration (CUDA)
  
- **NumPy** & **Pandas** - Data manipulation and numerical operations
  - Dataset manifest loading and processing
  - Metric calculations and statistics

- **Matplotlib** - Data visualization
  - Training curves and loss plots
  - Heatmap visualization
  - Prediction overlay visualization

- **Pillow (PIL)** - Image I/O and annotation

### DL Model Architecture Components

**WaferMatchNet** - Custom CNN for reference-to-search localization
- **Backbone options:**
  - Pre-trained **ResNet18** (ImageNet weights) - stem + layer1 (downsamples /4)
  - Fallback: From-scratch encoder (if pretrained unavailable)
  
- **Core Blocks:**
  - `ConvBlockGN` - Convolutional blocks with **Group Normalization** (instead of BatchNorm)
  - Separate branches for reference and search images
  - Feature extraction layers
  
- **Projection Head:**
  - Outputs 2D heatmap logits (correlation map)
  - Soft attention mechanism for coordinate regression
  
- **Loss Functions:**
  - Coordinate regression loss (normalized GT coordinates)
  - Heatmap cross-entropy loss
  - Weighted loss combination

### Training Infrastructure
- **Optimizer:** Adam with differential learning rates
  - Backbone LR: 5e-5 (cautious for pretrained features)
  - Head LR: 1e-3 (faster for new projection head)
  - L2 weight decay: 1e-5
  
- **Scheduling:**
  - Linear warmup: 5 epochs
  - Cosine annealing decay over remaining epochs
  - Early stopping: 30-epoch patience
  
- **Data Pipeline:**
  - Custom `WaferMatchDataset` class
  - Grayscale image loading (PIL)
  - Brightness augmentation (15% ±)
  - Bilinear resizing (ref: 100×100, search: 1000×1000)
  - Normalization to [0, 1] range

### Evaluation Metrics
- **Pixel Error (px_err):** L2 distance between predicted and GT coordinates
- **Peak Probability:** Heatmap maximum activation
- **Accuracy @ 5px:** Fraction of predictions within 5px of GT
- **Correlation-based heatmap visualization**

### Advanced Features
- **Soft ArgMax 2D** - Differentiable coordinate extraction from heatmaps
- **Heatmap generation** - Gaussian-blurred GT coordinates (σ=1.2)
- **Interactive Test Bench** - IPyWidgets UI for per-sample prediction
- **Checkpoint management:**
  - State dict saving/loading
  - Optimizer state preservation
  - Training history logging
  - Resume-from-checkpoint with fresh optimizer option
  
- **Visualization:**
  - Ground-truth vs. predicted box overlays
  - Raw correlation heatmaps
  - Training loss curves
  - Validation metrics plots

### Dataset Configuration
- **Multi-source:**
  - Mixed wafer dataset (varied architecture styles)
  - Mesh-only wafer dataset (fine crosshatch only)
  - Combined 2000+ training samples
  
- **Data splits:**
  - Train: 70%
  - Validation: 15%
  - Test: 15%

### Hyperparameters Summary
| Parameter | Value |
|-----------|-------|
| Batch Size | 6 |
| Epochs | 250 (with early stopping) |
| Warmup | 5 epochs |
| Early Stop Patience | 30 epochs |
| Heatmap Sigma | 1.2 |
| Coord Loss Weight | 2.0 |
| Accuracy Threshold | 5px |
| Input Resolutions | Ref: 100×100, Search: 1000×1000 |

---
## *Note* ##- Hyperparameters were changed during execution to meet the requirements of the model.
## Environment Requirements

### Runtime Environment
- **Python:** 3.12.13 (from notebook metadata)
- **GPU:** NVIDIA CUDA (Kaggle GPU accelerator)
- **Platform:** Kaggle Notebooks with Internet access enabled

### Python Package Versions (Inferred)
- torch (GPU-enabled)
- numpy
- pandas
- matplotlib
- pillow
- ipywidgets (for interactive notebook UI)

### Key Dependencies
- CUDA/cuDNN (for GPU acceleration)
- ImageNet pretrained weights (downloaded on first use)

---

## Workflow Integration

```
kaggle_1.py (Image Generator)
    ↓
    Generate heterogeneous SEM wafer pairs
    ↓
    Output: .png images + manifest.csv + .zip archive
    ↓
ray-tracer-v3.ipynb (DL Training)
    ↓
    Load dataset from .zip
    ↓
    Load Checkpoint.pt  (for revising the model with its values )
    ↓
    Train/Val/Test split
    ↓
    Train WaferMatchNet (250 epochs max)
    ↓
    Evaluate on test set
    ↓
    Fine-tune if needed (re-runnable)
    ↓
    Output: checkpoint_emergency.pt + metrics/graphs
```

---

## Summary Table

| Aspect | Image Generator | DL Model |
|--------|-----------------|----------|
| **Primary Framework** | PyTorch + NumPy + PIL | PyTorch |
| **Specialization** | Procedural SEM image synthesis | CNN-based spatial localization |
| **Key Innovation** | Heterogeneous die patterns + seeded reproducibility | Pretrained ResNet18 + soft attention heatmap |
| **Output Type** | Synthetic dataset (PNG + CSV) | Trained model checkpoint + metrics |
| **Compute** | CPU-friendly (synthesis) | GPU-optimized (CUDA) |
| **Data Format** | Wafer tiles with architecture variation | Reference-search image pairs |
| **Loss/Metric** | N/A (generation) | L2 pixel error + cross-entropy heatmap loss |

## Images 
### Training and Validation Graphs- 
<img width="1690" height="1300" alt="training_results" src="https://github.com/user-attachments/assets/9bbb5310-4867-48c3-b44d-ebcde5eff401" />

### Test Case Output - 
<img width="1479" height="511" alt="generated_image_cell10" src="https://github.com/user-attachments/assets/a8e5ff95-938f-42d5-9219-3d50e025353a" />
