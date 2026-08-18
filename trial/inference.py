#!/usr/bin/env python3
"""
=============================================================================
WAFER DEFECT LOCALIZATION - INFERENCE SERVER
=============================================================================
Production-ready inference for hackathon judges.

Usage:
  1. Place checkpoint at your configured path
  2. Run directly: python inference_app.py
     OR pass via arguments: python inference_app.py --ref reference.png --search search.png
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# MODEL ARCHITECTURE (identical to notebook)
# =============================================================================

class WaferMatchNet(nn.Module):
    """
    Siamese-inspired encoder + projection head.
    Takes two image halves (reference + search area) and predicts defect position.
    """
    def __init__(self, use_pretrained=True):
        super().__init__()
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            try:
                import torchvision.models as models
                resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.encoder = nn.Sequential(
                    resnet18.conv1,
                    resnet18.bn1,
                    resnet18.relu,
                    resnet18.maxpool,
                    resnet18.layer1,
                )
            except Exception as e:
                print(f"⚠ Pretrained backbone failed ({e}), falling back to from-scratch")
                self.encoder = self._build_encoder()
        else:
            self.encoder = self._build_encoder()
        
        # Head projects flattened features to heatmap
        self.head = nn.Sequential(
            nn.Linear(128 * 125 * 125, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1000 * 1000),  # output heatmap size
        )
    
    def _build_encoder(self):
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, ref, search):
        """Forward pass for reference (100px) and search (1000px) images."""
        ref_feat = self.encoder(ref)      # [B, 128, 25, 25]
        search_feat = self.encoder(search)  # [B, 128, 250, 250]
        
        # Global average pool
        ref_pooled = F.adaptive_avg_pool2d(ref_feat, (1, 1)).view(ref.size(0), -1)
        
        # Concatenate and project
        combined = torch.cat([ref_pooled, search_feat.view(search.size(0), -1)], dim=1)
        heatmap = self.head(combined).view(-1, 1, 1000, 1000)
        
        return heatmap


# =============================================================================
# INFERENCE PIPELINE
# =============================================================================

class WaferInference:
    def __init__(self, checkpoint_path: str, device: str = "auto"):
        """
        Load model and checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint
            device: "cuda", "cpu", or "auto" (default)
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] Using device: {self.device}")
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = WaferMatchNet().to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        
        # Extract metadata
        self.history = ckpt.get("history", {})
        self.epoch = ckpt.get("epoch", "unknown")
        
        # Best performance
        val_errs = self.history.get("val_px_err", [])
        self.best_val_err = min(val_errs) if val_errs else None
        
        print(f"[INFO] Model loaded from epoch {self.epoch}")
        if self.best_val_err:
            print(f"[INFO] Best validation error: {self.best_val_err:.2f}px")
    
    def preprocess(self, image_path: str, target_size: int) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to input image
            target_size: Target size (100 for ref, 1000 for search)
        
        Returns:
            Normalized tensor [1, 1, H, W]
        """
        img = Image.open(image_path).convert("L")
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Normalize to [-1, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, ref_path: str, search_path: str) -> Dict[str, Any]:
        """
        Predict defect position.
        
        Args:
            ref_path: Path to reference image (typically 100x100)
            search_path: Path to search image (typically 1000x1000)
        
        Returns:
            Dict with prediction info and coordinates
        """
        # Validate paths
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference image not found: {ref_path}")
        if not os.path.exists(search_path):
            raise FileNotFoundError(f"Search image not found: {search_path}")
        
        print(f"\n[INFERENCE]")
        print(f"  Reference: {ref_path}")
        print(f"  Search:    {search_path}")
        
        # Preprocess
        ref_tensor = self.preprocess(ref_path, 100)
        search_tensor = self.preprocess(search_path, 1000)
        
        # Forward pass
        with torch.no_grad():
            heatmap = self.model(ref_tensor, search_tensor)  # [1, 1, 1000, 1000]
        
        heatmap_np = heatmap.squeeze().cpu().numpy()
        
        # Extract peak
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        confidence = float(heatmap_np[y_idx, x_idx])
        
        return {
            "x": float(x_idx),
            "y": float(y_idx),
            "confidence": confidence,
            "heatmap": heatmap_np,
            "metadata": {
                "model_epoch": self.epoch,
                "best_val_error_px": self.best_val_err,
                "device": str(self.device),
                "input_shapes": {
                    "reference": (100, 100),
                    "search": (1000, 1000),
                }
            }
        }
    
    def visualize_result(self, search_path: str, x: float, y: float, 
                        output_path: str = "result_visualization.png", 
                        radius: int = 20):
        """
        Draw prediction on search image with a circle + crosshair.
        """
        img = Image.open(search_path).convert("L")
        img = img.resize((1000, 1000), Image.Resampling.LANCZOS)
        img_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)
        
        # Draw circle
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline="red", width=3
        )
        
        # Draw crosshair
        line_len = radius * 1.5
        draw.line([(x - line_len, y), (x + line_len, y)], fill="red", width=2)
        draw.line([(x, y - line_len), (x, y + line_len)], fill="red", width=2)
        
        # Draw text
        draw.text((x + radius + 5, y - 20), f"({x:.0f}, {y:.0f})", fill="white")
        
        img_rgb.save(output_path)
        print(f"[SAVED] Visualization: {output_path}")
        return output_path


# =============================================================================
# CLI & MAIN
# =============================================================================

def main():

    # ==========================================================================
    # MANUAL PATH CONFIGURATION — Change folder/filenames here
    # ==========================================================================
    BASE_DIR = r"D:\Coding shit\Python\img_gen\trial"

    CHECKPOINT_PATH = os.path.join(BASE_DIR, "personal_best_checkpoint.pt")
    REF_PATH        = os.path.join(BASE_DIR, "reference.png")
    SEARCH_PATH     = os.path.join(BASE_DIR, "search.png")
    OUTPUT_VIZ      = os.path.join(BASE_DIR, "result_visualization.png")
    OUTPUT_JSON     = os.path.join(BASE_DIR, "result.json")
    # ==========================================================================

    parser = argparse.ArgumentParser(
        description="Wafer Defect Localization Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                       help="Path to model checkpoint")
    parser.add_argument("--ref", type=str, nargs="+", default=[REF_PATH],
                       help="Reference image path(s)")
    parser.add_argument("--search", type=str, nargs="+", default=[SEARCH_PATH],
                       help="Search image path(s)")
    parser.add_argument("--json", type=str, default=OUTPUT_JSON,
                       help="Save results as JSON (optional)")
    parser.add_argument("--visualize", type=str, nargs="+", default=[OUTPUT_VIZ],
                       help="Save visualizations with predicted circles")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'cuda', 'cpu', or 'auto' (default)")
    
    args = parser.parse_args()
    
    # Ensure matching counts
    if len(args.ref) != len(args.search):
        raise ValueError(f"Mismatch: {len(args.ref)} reference images but {len(args.search)} search images")
    
    # Initialize inference engine
    try:
        engine = WaferInference(args.checkpoint, device=args.device)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    # Run predictions
    results = []
    visualizations = []
    
    for i, (ref_path, search_path) in enumerate(zip(args.ref, args.search)):
        try:
            result = engine.predict(ref_path, search_path)
            results.append(result)
            
            # Print result
            print(f"\n[RESULT {i+1}]")
            print(f"  Position:  x={result['x']:.1f}, y={result['y']:.1f}")
            print(f"  Confidence: {result['confidence']:.4f}")
            
            # Visualize if requested
            if args.visualize:
                viz_path = args.visualize[i] if i < len(args.visualize) else f"result_{i+1}.png"
                engine.visualize_result(search_path, result["x"], result["y"], viz_path)
                visualizations.append(viz_path)
        
        except Exception as e:
            print(f"[ERROR] Processing pair {i+1}: {e}")
            results.append({"error": str(e)})
    
    # Save JSON if requested
    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "results": results,
                "num_predictions": len(results),
            }, f, indent=2)
        print(f"\n[SAVED] JSON results: {args.json}")
    
    # Summary
    print("\n" + "="*60)
    print(f"[SUMMARY] Processed {len(results)} predictions")
    print(f"  Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"  Failed:     {sum(1 for r in results if 'error' in r)}")
    if visualizations:
        print(f"  Visualizations: {', '.join(visualizations)}")
    print("="*60)


if __name__ == "__main__":
    main()
