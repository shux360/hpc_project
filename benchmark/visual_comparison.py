#!/usr/bin/env python3
"""
Visual Accuracy Comparison Tool
Generates side-by-side comparisons and difference maps
"""

import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Tuple

class VisualComparator:
    def __init__(self, base_path: str = "./", output_dir: str = "./accuracy_comparison"):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_image(self, method: str, stage: str) -> np.ndarray:
        """Load image"""
        filename = f"{method}_{stage}.png"
        filepath = self.base_path / filename
        
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Not found: {filepath}")
            return None
        return img
    
    def create_difference_map(self, serial: np.ndarray, parallel: np.ndarray) -> np.ndarray:
        """Create color difference map"""
        diff = np.abs(serial.astype(np.float32) - parallel.astype(np.float32))
        
        # Normalize to 0-255 for visualization
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
        
        # Apply colormap (red = high difference, blue = low difference)
        diff_color = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
        
        return diff_color
    
    def create_side_by_side(self, serial: np.ndarray, parallel: np.ndarray, 
                          diff_map: np.ndarray) -> np.ndarray:
        """Create 3-panel comparison image"""
        height, width = serial.shape
        
        # Convert grayscale to BGR for consistent 3-channel output
        serial_bgr = cv2.cvtColor(serial, cv2.COLOR_GRAY2BGR)
        parallel_bgr = cv2.cvtColor(parallel, cv2.COLOR_GRAY2BGR)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        cv2.putText(serial_bgr, "Serial (Reference)", (10, 30), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(parallel_bgr, "Parallel Implementation", (10, 30), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(diff_map, "Difference Map", (10, 30), font, font_scale, (255, 255, 255), thickness)
        
        # Stack horizontally
        comparison = np.hstack([serial_bgr, parallel_bgr, diff_map])
        
        return comparison
    
    def create_statistics_image(self, serial: np.ndarray, parallel: np.ndarray,
                               method: str, stage: str) -> Tuple[np.ndarray, dict]:
        """Create image with statistics overlay"""
        diff = np.abs(serial.astype(np.float32) - parallel.astype(np.float32))
        
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(diff)
        max_diff = np.max(diff)
        different_pixels = np.count_nonzero(diff)
        total_pixels = diff.size
        
        stats = {
            "rmse": rmse,
            "mae": mae,
            "max": max_diff,
            "different": different_pixels,
            "total": total_pixels
        }
        
        # Create white background for text
        height = 200
        width = 600
        stats_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        y_offset = 30
        line_height = 25
        
        texts = [
            f"Method: {method.upper()} - {stage.upper()}",
            f"RMSE: {rmse:.6f}",
            f"MAE: {mae:.6f}",
            f"Max Difference: {max_diff:.2f}",
            f"Different Pixels: {different_pixels} / {total_pixels}",
            f"Match: {((total_pixels - different_pixels) / total_pixels * 100):.4f}%"
        ]
        
        for text in texts:
            cv2.putText(stats_img, text, (20, y_offset), font, font_scale, (0, 0, 0), thickness)
            y_offset += line_height
        
        return stats_img, stats
    
    def generate_comparison_images(self, method: str, stage: str):
        """Generate all comparison images for a method"""
        print(f"\nProcessing {method.upper()} - {stage.upper()}...")
        
        serial = self.load_image("serial", stage)
        parallel = self.load_image(method, stage)
        
        if serial is None or parallel is None:
            print(f"  [SKIP] Missing images")
            return False
        
        if serial.shape != parallel.shape:
            print(f"  [ERROR] Shape mismatch")
            return False
        
        # Generate visualizations
        diff_map = self.create_difference_map(serial, parallel)
        comparison = self.create_side_by_side(serial, parallel, diff_map)
        stats_img, stats = self.create_statistics_image(serial, parallel, method, stage)
        
        # Save outputs
        output_prefix = f"{method}_{stage}"
        
        cv2.imwrite(str(self.output_dir / f"{output_prefix}_comparison.png"), comparison)
        cv2.imwrite(str(self.output_dir / f"{output_prefix}_difference_map.png"), diff_map)
        cv2.imwrite(str(self.output_dir / f"{output_prefix}_statistics.png"), stats_img)
        
        print(f"  Generated 3 images:")
        print(f"     - {output_prefix}_comparison.png (side-by-side)")
        print(f"     - {output_prefix}_difference_map.png (heatmap)")
        print(f"     - {output_prefix}_statistics.png (metrics)")
        
        return True
    
    def run_visual_comparison(self):
        """Run full visual comparison"""
        print("="*70)
        print("VISUAL ACCURACY COMPARISON TOOL")
        print("Generating difference maps and side-by-side comparisons")
        print("="*70)
        
        methods = ["openmp", "pthread", "mpi", "hybrid"]
        stages = ["denoised", "edges"]
        
        success_count = 0
        for method in methods:
            for stage in stages:
                if self.generate_comparison_images(method, stage):
                    success_count += 1
        
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Generated {success_count} comparison image sets")
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"\nGenerated files:")
        print(f"  - *_comparison.png: 3-panel side-by-side view")
        print(f"  - *_difference_map.png: Heatmap of pixel differences")
        print(f"  - *_statistics.png: Accuracy metrics")
        
        return success_count > 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visual accuracy comparisons")
    parser.add_argument("--path", default="./", 
                       help="Base path for image files")
    parser.add_argument("--out", default="./accuracy_comparison",
                       help="Directory for generated comparison images")
    
    args = parser.parse_args()
    
    comparator = VisualComparator(args.path, args.out)
    success = comparator.run_visual_comparison()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
