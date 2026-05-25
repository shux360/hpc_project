#!/usr/bin/env python3
"""
HPC Image Processing Accuracy Comparison Tool
Compares parallel implementations against serial baseline using RMSE, MAE, SSIM
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional
import sys

@dataclass
class ComparisonMetrics:
    rmse: float
    mae: float
    max_diff: float
    min_diff: float
    different_pixels: int
    total_pixels: int
    similarity_percent: float
    ssim: float  # Structural Similarity Index

class ImageComparator:
    def __init__(self, base_path: str = "./"):
        self.base_path = Path(base_path)
        self.methods = ["serial", "openmp", "pthread", "mpi", "hybrid"]
        self.stages = ["denoised", "edges"]
    
    def load_image(self, method: str, stage: str) -> np.ndarray:
        """Load image for a specific method and stage"""
        filename = f"{method}_{stage}.png"
        filepath = self.base_path / filename
        
        img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] Image not found: {filepath}")
            return None
        return img
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Convert to normalized float and use SSIM constants for the 0.0-1.0 scale.
        img1_f = img1.astype(np.float64) / 255.0
        img2_f = img2.astype(np.float64) / 255.0
        
        # Calculate mean
        mu1 = cv2.blur(img1_f, (11, 11))
        mu2 = cv2.blur(img2_f, (11, 11))
        
        # Calculate variance and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.blur(img1_f ** 2, (11, 11)) - mu1_sq
        sigma2_sq = cv2.blur(img2_f ** 2, (11, 11)) - mu2_sq
        sigma12 = cv2.blur(img1_f * img2_f, (11, 11)) - mu1_mu2
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return np.mean(ssim_map)
    
    def compare_images(self, serial: np.ndarray, parallel: np.ndarray,
                      method_name: str, stage_name: str) -> Optional[ComparisonMetrics]:
        """Compare serial and parallel images"""
        
        if serial is None or parallel is None:
            print(f"[ERROR] Missing image for {method_name} {stage_name}")
            return None
        
        if serial.shape != parallel.shape:
            print(f"[ERROR] Shape mismatch for {method_name} {stage_name}")
            print(f"  Serial: {serial.shape}, Parallel: {parallel.shape}")
            return None
        
        total_pixels = serial.size
        diff = np.abs(serial.astype(np.float32) - parallel.astype(np.float32))
        
        rmse = np.sqrt(np.mean(diff ** 2))
        mae = np.mean(diff)
        max_diff = np.max(diff)
        min_diff = np.min(diff)
        different_pixels = np.count_nonzero(diff)
        similarity_percent = ((total_pixels - different_pixels) / total_pixels) * 100
        
        ssim = self.calculate_ssim(serial, parallel)
        
        return ComparisonMetrics(
            rmse=rmse,
            mae=mae,
            max_diff=max_diff,
            min_diff=min_diff,
            different_pixels=different_pixels,
            total_pixels=total_pixels,
            similarity_percent=similarity_percent,
            ssim=ssim
        )
    
    def print_comparison_result(self, metrics: ComparisonMetrics, 
                               method: str, stage: str):
        """Print comparison results in formatted table"""
        print(f"\n{'='*70}")
        print(f"METHOD: {method.upper()} - {stage.upper()}")
        print(f"{'='*70}")
        
        print(f"  RMSE (Root Mean Square Error):  {metrics.rmse:.6f}")
        print(f"  MAE  (Mean Absolute Error):     {metrics.mae:.6f}")
        print(f"  SSIM (Structural Similarity):   {metrics.ssim:.6f}")
        print(f"  Max Difference:                 {metrics.max_diff:.2f}")
        print(f"  Min Difference:                 {metrics.min_diff:.2f}")
        print(f"  Different Pixels:               {metrics.different_pixels} / {metrics.total_pixels}")
        print(f"  Similarity to Serial:           {metrics.similarity_percent:.4f}%")
        
        print(f"\n  Status: ", end="")
        if metrics.rmse < 0.001 and metrics.different_pixels == 0:
            print("PERFECT MATCH (Identical to serial)")
        elif metrics.rmse < 0.5 and metrics.similarity_percent > 99.99:
            print("EXCELLENT (Negligible difference)")
        elif metrics.rmse < 1.0 and metrics.similarity_percent > 99.0:
            print("ACCEPTABLE (Minor rounding differences)")
        else:
            print("FAILED (Significant difference)")
    
    def run_full_comparison(self):
        """Run full accuracy comparison"""
        print("\n" + "="*70)
        print("HPC IMAGE PROCESSING - ACCURACY COMPARISON TOOL")
        print("Comparing all parallel implementations against serial baseline")
        print("="*70)
        
        # Load serial baseline
        serial_denoised = self.load_image("serial", "denoised")
        serial_edges = self.load_image("serial", "edges")
        
        if serial_denoised is None or serial_edges is None:
            print("[ERROR] Serial baseline files not found")
            return False
        
        print(f"\n[INFO] Serial baseline loaded")
        print(f"  Denoised: {serial_denoised.shape} pixels")
        print(f"  Edges: {serial_edges.shape} pixels")
        
        # Store results for summary
        denoised_results: Dict[str, ComparisonMetrics] = {}
        edges_results: Dict[str, ComparisonMetrics] = {}
        
        # Compare denoising stage
        print("\n" + "="*70)
        print("DENOISING STAGE COMPARISON (Gaussian Blur)")
        print("="*70)
        
        for method in self.methods:
            parallel_denoised = self.load_image(method, "denoised")
            if parallel_denoised is not None:
                metrics = self.compare_images(serial_denoised, parallel_denoised, 
                                            method, "denoised")
                if metrics:
                    self.print_comparison_result(metrics, method, "Denoised")
                    denoised_results[method] = metrics
        
        # Compare edge detection stage
        print("\n" + "="*70)
        print("EDGE DETECTION STAGE COMPARISON (Sobel Operator)")
        print("="*70)
        
        for method in self.methods:
            parallel_edges = self.load_image(method, "edges")
            if parallel_edges is not None:
                metrics = self.compare_images(serial_edges, parallel_edges, 
                                            method, "edges")
                if metrics:
                    self.print_comparison_result(metrics, method, "Edges")
                    edges_results[method] = metrics
        
        # Print summary tables
        self._print_summary_table("DENOISING", denoised_results)
        self._print_summary_table("EDGE DETECTION", edges_results)
        
        # Overall conclusion
        self._print_conclusion(edges_results)
        
        return True
    
    def _print_summary_table(self, stage_name: str, results: Dict[str, ComparisonMetrics]):
        """Print summary table for a stage"""
        if not results:
            return
        
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE - {stage_name}")
        print(f"{'='*80}")
        
        header = f"{'Method':<12} {'RMSE':<12} {'MAE':<12} {'SSIM':<12} {'Match %':<15} {'Status':<15}"
        print(header)
        print("-" * 80)
        
        for method in self.methods:
            if method in results:
                r = results[method]
                
                status = ""
                if r.rmse < 0.001:
                    status = "Perfect"
                elif r.rmse < 0.5:
                    status = "Excellent"
                elif r.rmse < 1.0:
                    status = "Acceptable"
                else:
                    status = "Failed"
                
                print(f"{method:<12} {r.rmse:<12.6f} {r.mae:<12.6f} {r.ssim:<12.6f} "
                      f"{r.similarity_percent:<14.4f}% {status:<15}")
    
    def _print_conclusion(self, edges_results: Dict[str, ComparisonMetrics]):
        """Print overall conclusion"""
        print(f"\n{'='*70}")
        print("CONCLUSION")
        print(f"{'='*70}")
        
        if not edges_results:
            print("No comparison results available")
            return
        
        all_perfect = all(r.rmse < 0.001 and r.different_pixels == 0 
                         for r in edges_results.values())
        all_excellent = all(r.rmse < 0.5
                           for r in edges_results.values())
        
        if all_perfect:
            print("ALL IMPLEMENTATIONS MATCH SERIAL PERFECTLY")
            print("   All parallel versions produce identical edge detection results.")
            print("   (RMSE < 0.001 and zero different pixels)")
        elif all_excellent:
            print("ALL IMPLEMENTATIONS MATCH SERIAL EXCELLENTLY")
            print("   All parallel versions produce nearly identical results.")
            print("   (RMSE < 0.5 - differences negligible)")
        else:
            print("SOME IMPLEMENTATIONS SHOW DIFFERENCES")
            print("   Review results above for details.")
            
            for method, metrics in edges_results.items():
                if metrics.rmse >= 0.5:
                    print(f"   - {method}: RMSE = {metrics.rmse:.6f}")
        
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare parallel implementations accuracy")
    parser.add_argument("--path", default="./", 
                       help="Base path for image files (default: ./)")
    
    args = parser.parse_args()
    
    comparator = ImageComparator(args.path)
    success = comparator.run_full_comparison()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
