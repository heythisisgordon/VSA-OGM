"""Integration tests for the VSA-OGM implementation."""

import os
import sys
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from src directory
from src.main import pointcloud_to_ogm
from src.io import load_pointcloud, save_occupancy_grid, convert_occupancy_grid_to_pointcloud
from src.utils import visualize_occupancy_grid, visualize_class_grid

def test_full_pipeline():
    """Test the full VSA-OGM pipeline."""
    print("Testing full VSA-OGM pipeline...")
    
    # Create a small test grid
    grid = np.zeros((20, 20), dtype=np.int32)
    grid[8:12, 8:12] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 2, 0, 2]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Save point cloud to temporary file
    temp_input = "temp_input.npy"
    temp_output = "temp_output.npz"
    
    combined = np.column_stack((points.numpy(), labels.numpy()))
    np.save(temp_input, combined)
    
    try:
        # Run the full pipeline
        result = pointcloud_to_ogm(
            input_file=temp_input,
            output_file=temp_output,
            world_bounds=world_bounds,
            resolution=resolution,
            vsa_dimensions=1000,  # Small for testing
            use_cuda=False,
            verbose=True,
            incremental=True,
            horizon_distance=0.5,
            sample_resolution=0.2,
            max_samples=10,
            safety_margin=0.1,
            occupied_disk_radius=2,
            empty_disk_radius=4,
            save_entropy_grids=True,
            save_stats=True,
            visualize=True
        )
        
        # Check result
        assert "grid" in result
        assert "class_grid" in result
        assert "occupied_entropy" in result
        assert "empty_entropy" in result
        assert "global_entropy" in result
        assert "stats" in result
        
        print("Result contains all expected keys")
        
        # Check output file exists
        assert os.path.exists(temp_output)
        print(f"Output file {temp_output} created successfully")
        
        # Check visualization files exist
        visualization_files = [
            os.path.splitext(temp_output)[0] + "_visualization.png",
            os.path.splitext(temp_output)[0] + "_class.png",
            os.path.splitext(temp_output)[0] + "_occupied_entropy.png",
            os.path.splitext(temp_output)[0] + "_empty_entropy.png",
            os.path.splitext(temp_output)[0] + "_global_entropy.png",
            os.path.splitext(temp_output)[0] + "_entropy_comparison.png"
        ]
        
        for file in visualization_files:
            assert os.path.exists(file)
            print(f"Visualization file {file} created successfully")
        
        print("Full pipeline test passed!")
        return True
    finally:
        # Clean up temporary files
        for file in [temp_input, temp_output]:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed temporary file {file}")
        
        # Clean up visualization files
        for file in visualization_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed visualization file {file}")

def test_incremental_parameters():
    """Test incremental processing with different parameters."""
    print("Testing incremental processing with different parameters...")
    
    # Create a small test grid
    grid = np.zeros((20, 20), dtype=np.int32)
    grid[8:12, 8:12] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 2, 0, 2]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Save point cloud to temporary file
    temp_input = "temp_input.npy"
    
    combined = np.column_stack((points.numpy(), labels.numpy()))
    np.save(temp_input, combined)
    
    try:
        # Test different horizon distances
        horizon_distances = [0.3, 0.5, 1.0]
        
        for horizon in horizon_distances:
            temp_output = f"temp_output_horizon_{horizon}.npz"
            
            print(f"\nTesting with horizon_distance={horizon}")
            
            # Run the pipeline
            result = pointcloud_to_ogm(
                input_file=temp_input,
                output_file=temp_output,
                world_bounds=world_bounds,
                resolution=resolution,
                vsa_dimensions=1000,  # Small for testing
                use_cuda=False,
                verbose=False,
                incremental=True,
                horizon_distance=horizon,
                sample_resolution=0.2,
                max_samples=10,
                safety_margin=0.1
            )
            
            # Check result
            assert "grid" in result
            assert "stats" in result
            
            # Check stats
            stats = result["stats"]
            assert "incremental_time" in stats
            assert "total_samples_processed" in stats
            
            print(f"Processed {stats['total_samples_processed']} samples in {stats['incremental_time']:.4f} seconds")
            
            # Clean up output file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        # Test different sample resolutions
        sample_resolutions = [0.1, 0.2, 0.5]
        
        for sample_res in sample_resolutions:
            temp_output = f"temp_output_sample_res_{sample_res}.npz"
            
            print(f"\nTesting with sample_resolution={sample_res}")
            
            # Run the pipeline
            result = pointcloud_to_ogm(
                input_file=temp_input,
                output_file=temp_output,
                world_bounds=world_bounds,
                resolution=resolution,
                vsa_dimensions=1000,  # Small for testing
                use_cuda=False,
                verbose=False,
                incremental=True,
                horizon_distance=0.5,
                sample_resolution=sample_res,
                max_samples=10,
                safety_margin=0.1
            )
            
            # Check result
            assert "grid" in result
            assert "stats" in result
            
            # Check stats
            stats = result["stats"]
            assert "incremental_time" in stats
            assert "total_samples_processed" in stats
            
            print(f"Processed {stats['total_samples_processed']} samples in {stats['incremental_time']:.4f} seconds")
            
            # Clean up output file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        print("\nIncremental parameters test passed!")
        return True
    finally:
        # Clean up input file
        if os.path.exists(temp_input):
            os.remove(temp_input)
            print(f"Removed temporary file {temp_input}")

def test_entropy_parameters():
    """Test Shannon entropy feature extraction with different parameters."""
    print("Testing Shannon entropy feature extraction with different parameters...")
    
    # Create a small test grid
    grid = np.zeros((20, 20), dtype=np.int32)
    grid[8:12, 8:12] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 2, 0, 2]
    resolution = 0.1
    
    points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
    
    # Save point cloud to temporary file
    temp_input = "temp_input.npy"
    
    combined = np.column_stack((points.numpy(), labels.numpy()))
    np.save(temp_input, combined)
    
    try:
        # Test different disk radii combinations
        disk_radii_combinations = [
            (1, 2),  # (occupied, empty)
            (2, 4),
            (3, 6)
        ]
        
        results = {}
        
        for occupied_radius, empty_radius in disk_radii_combinations:
            temp_output = f"temp_output_occ{occupied_radius}_emp{empty_radius}.npz"
            
            print(f"\nTesting with occupied_radius={occupied_radius}, empty_radius={empty_radius}")
            
            # Run the pipeline
            result = pointcloud_to_ogm(
                input_file=temp_input,
                output_file=temp_output,
                world_bounds=world_bounds,
                resolution=resolution,
                vsa_dimensions=1000,  # Small for testing
                use_cuda=False,
                verbose=False,
                incremental=False,
                occupied_disk_radius=occupied_radius,
                empty_disk_radius=empty_radius,
                save_entropy_grids=True
            )
            
            # Check result
            assert "occupied_entropy" in result
            assert "empty_entropy" in result
            assert "global_entropy" in result
            
            # Get entropy grids
            occupied_entropy = result["occupied_entropy"]
            empty_entropy = result["empty_entropy"]
            global_entropy = result["global_entropy"]
            
            # Check shapes
            assert occupied_entropy.shape == (20, 20)
            assert empty_entropy.shape == (20, 20)
            assert global_entropy.shape == (20, 20)
            
            # Check that entropy values are within expected range [0, 1]
            assert torch.all(occupied_entropy >= 0) and torch.all(occupied_entropy <= 1)
            assert torch.all(empty_entropy >= 0) and torch.all(empty_entropy <= 1)
            
            # Get class grid
            class_grid = result["class_grid"]
            
            # Count class distribution
            occupied_count = torch.sum(class_grid == 1).item()
            empty_count = torch.sum(class_grid == -1).item()
            unknown_count = torch.sum(class_grid == 0).item()
            total_count = class_grid.numel()
            
            occupied_percent = 100 * occupied_count / total_count
            empty_percent = 100 * empty_count / total_count
            unknown_percent = 100 * unknown_count / total_count
            
            print(f"Class distribution:")
            print(f"  - Occupied: {occupied_count} ({occupied_percent:.2f}%)")
            print(f"  - Empty: {empty_count} ({empty_percent:.2f}%)")
            print(f"  - Unknown: {unknown_count} ({unknown_percent:.2f}%)")
            
            # Store results
            results[(occupied_radius, empty_radius)] = {
                "occupied_count": occupied_count,
                "empty_count": empty_count,
                "unknown_count": unknown_count,
                "occupied_percent": occupied_percent,
                "empty_percent": empty_percent,
                "unknown_percent": unknown_percent
            }
            
            # Clean up output file
            if os.path.exists(temp_output):
                os.remove(temp_output)
        
        # Print comparison summary
        print("\n=== Shannon Entropy Parameter Comparison ===\n")
        print("| Occupied Radius | Empty Radius | Occupied % | Empty % | Unknown % |")
        print("|----------------|--------------|------------|---------|-----------|")
        
        for (occupied_radius, empty_radius), data in results.items():
            print(f"| {occupied_radius:14d} | {empty_radius:12d} | {data['occupied_percent']:10.2f}% | {data['empty_percent']:7.2f}% | {data['unknown_percent']:9.2f}% |")
        
        print("\nEntropy parameters test passed!")
        return True
    finally:
        # Clean up input file
        if os.path.exists(temp_input):
            os.remove(temp_input)
            print(f"Removed temporary file {temp_input}")

def run_all_tests():
    """Run all integration tests."""
    tests = [
        test_full_pipeline,
        test_incremental_parameters,
        test_entropy_parameters
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f"{test.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"{test.__name__}: FAILED with exception: {e}")
            results.append(False)
    
    # Print summary
    print("\nIntegration Test Summary:")
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
