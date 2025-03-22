# Phase 4: Main Interface and CLI Updates

## Summary of Overall Task

The overall task is to implement a Grid-Based Sequential VSA-OGM mapping pipeline that treats each sampling point like a robot pose keyframe. This approach processes the point cloud incrementally, focusing on local points within sensor range at each sampling location, rather than processing the entire point cloud at once. The implementation will include:

1. Efficient spatial indexing with adaptive cell sizing (Phase 1)
2. Optimized vector computation with parallel processing and caching (Phase 2)
3. Memory-aware processing with GPU memory monitoring (Phase 3)
4. Incremental processing with horizon-limited visibility (Phase 3)
5. Enhanced VSA mapper with direct spatial processing (Phase 3)
6. Updated main interface and CLI (Phase 4 - Current)

## Phase 4 Focus: Main Interface and CLI Updates

In this phase, we will update the main interface and command-line interface (CLI) to support the enhanced VSA mapper and provide access to its new features. This will ensure that users can easily leverage the improved functionality through both the API and the command line.

### Current Implementation Analysis

The current main interface in `src/main.py` provides a function `pointcloud_to_ogm` and a CLI entry point `cli_main`, but they only support the original `VSAMapper` class. The interface needs to be updated to:
- Support both the original and enhanced mappers
- Provide access to the new features of the enhanced mapper
- Include options for performance monitoring and statistics
- Support visualization of results

### Implementation Plan

1. **Updated pointcloud_to_ogm Function**

```python
def pointcloud_to_ogm(
    input_file: str,
    output_file: str,
    world_bounds: Optional[List[float]] = None,
    resolution: float = 0.1,
    vsa_dimensions: int = 16000,
    use_cuda: bool = True,
    verbose: bool = False,
    incremental: bool = False,
    horizon_distance: float = 10.0,
    sample_resolution: Optional[float] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1000,
    length_scale: float = 2.0,
    min_cell_resolution: Optional[float] = None,
    max_cell_resolution: Optional[float] = None,
    cache_size: int = 10000,
    memory_threshold: float = 0.8,
    safety_margin: float = 0.5,
    use_enhanced_mapper: bool = False,
    save_stats: bool = False,
    visualize: bool = False
) -> Dict[str, Any]:
    """
    Convert a point cloud to an occupancy grid map.
    
    Args:
        input_file: Path to input point cloud (.npy file)
        output_file: Path to save output occupancy grid (.npz file)
        world_bounds: World bounds [x_min, x_max, y_min, y_max]
        resolution: Grid resolution in meters
        vsa_dimensions: Dimensionality of VSA vectors
        use_cuda: Whether to use CUDA if available
        verbose: Whether to print verbose output
        incremental: Whether to use incremental processing
        horizon_distance: Maximum distance from sample point to consider points
        sample_resolution: Resolution for sampling grid (default: 10x resolution)
        max_samples: Maximum number of sample positions to process
        batch_size: Batch size for processing points
        length_scale: Length scale for power operation
        min_cell_resolution: Minimum resolution for spatial indexing (default: 5x resolution)
        max_cell_resolution: Maximum resolution for spatial indexing (default: 20x resolution)
        cache_size: Maximum size of vector cache
        memory_threshold: Threshold for GPU memory usage (0.0-1.0)
        safety_margin: Minimum distance from occupied points for sampling
        use_enhanced_mapper: Whether to use the enhanced mapper
        save_stats: Whether to save processing statistics
        visualize: Whether to visualize results
        
    Returns:
        Dictionary with processing results and statistics
    """
    # Record start time
    start_time = time.time()
    
    # Set device (default to CUDA when available)
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading point cloud from {input_file}")
    
    # Load point cloud
    points, labels = io.load_pointcloud(input_file, device)
    
    if verbose:
        print(f"Loaded point cloud with {points.shape[0]} points")
    
    # Set default world bounds if not provided
    if world_bounds is None:
        # Calculate from point cloud with padding
        x_min, y_min = points.min(dim=0).values.cpu().numpy() - 5.0
        x_max, y_max = points.max(dim=0).values.cpu().numpy() + 5.0
        world_bounds = [float(x_min), float(x_max), float(y_min), float(y_max)]
        
        if verbose:
            print(f"Automatically determined world bounds: {world_bounds}")
    
    # Set default cell resolutions if not provided
    if min_cell_resolution is None:
        min_cell_resolution = resolution * 5
    
    if max_cell_resolution is None:
        max_cell_resolution = resolution * 20
    
    # Create mapper configuration
    config = {
        "world_bounds": world_bounds,
        "resolution": resolution,
        "min_cell_resolution": min_cell_resolution,
        "max_cell_resolution": max_cell_resolution,
        "vsa_dimensions": vsa_dimensions,
        "length_scale": length_scale,
        "decision_thresholds": [-0.99, 0.99],
        "verbose": verbose,
        "batch_size": batch_size,
        "cache_size": cache_size,
        "memory_threshold": memory_threshold
    }
    
    # Create appropriate mapper based on use_enhanced_mapper flag
    if use_enhanced_mapper:
        if verbose:
            print("Initializing EnhancedVSAMapper...")
        
        mapper = EnhancedVSAMapper(config, device=device)
    else:
        if verbose:
            print("Initializing VSAMapper...")
        
        mapper = VSAMapper(config, device=device)
    
    # Process point cloud
    if verbose:
        print("Processing point cloud...")
    
    mapper.process_observation(points, labels)
    
    # Process incrementally if requested
    if incremental:
        if verbose:
            print("Processing incrementally...")
        
        if use_enhanced_mapper:
            # Enhanced mapper supports safety margin
            mapper.process_incrementally(
                horizon_distance=horizon_distance,
                sample_resolution=sample_resolution,
                max_samples=max_samples,
                safety_margin=safety_margin
            )
        else:
            # Original mapper doesn't support safety margin
            mapper.process_incrementally(
                horizon_distance=horizon_distance,
                sample_resolution=sample_resolution,
                max_samples=max_samples
            )
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Get occupancy grid
    if verbose:
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Saving occupancy grid to {output_file}")
    
    grid = mapper.get_occupancy_grid()
    
    # Create metadata
    metadata = {
        "world_bounds": world_bounds, 
        "resolution": resolution,
        "incremental": incremental,
        "horizon_distance": horizon_distance if incremental else None,
        "sample_resolution": sample_resolution if incremental else None,
        "processing_time": processing_time,
        "use_enhanced_mapper": use_enhanced_mapper
    }
    
    # Add enhanced mapper statistics if available
    if use_enhanced_mapper:
        stats = mapper.get_stats()
        if save_stats:
            metadata["stats"] = stats
    
    # Save occupancy grid
    io.save_occupancy_grid(grid, output_file, metadata=metadata)
    
    # Visualize results if requested
    if visualize:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Visualize occupancy grid
        visualize_file = os.path.splitext(output_file)[0] + "_visualization.png"
        utils.visualize_occupancy_grid(
            grid=grid,
            output_file=visualize_file,
            world_bounds=world_bounds,
            colormap="viridis",
            show=False
        )
        
        if verbose:
            print(f"Visualization saved to {visualize_file}")
        
        # Visualize class grid if available
        try:
            class_grid = mapper.get_class_grid()
            class_file = os.path.splitext(output_file)[0] + "_class.png"
            utils.visualize_class_grid(
                grid=class_grid,
                output_file=class_file,
                world_bounds=world_bounds,
                show=False
            )
            
            if verbose:
                print(f"Class grid visualization saved to {class_file}")
        except:
            if verbose:
                print("Class grid visualization not available")
    
    if verbose:
        print(f"Occupancy grid saved to {output_file}")
    
    # Prepare result dictionary
    result = {
        "grid": grid,
        "metadata": metadata,
        "processing_time": processing_time
    }
    
    # Add enhanced mapper statistics if available
    if use_enhanced_mapper:
        result["stats"] = mapper.get_stats()
    
    return result
```

2. **Updated CLI Interface**

```python
def cli_main() -> None:
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Convert point cloud to occupancy grid map")
    parser.add_argument("input", help="Input point cloud file (.npy)")
    parser.add_argument("output", help="Output occupancy grid file (.npz)")
    
    # Basic options
    parser.add_argument("--bounds", "-b", nargs=4, type=float, help="World bounds [x_min x_max y_min y_max]")
    parser.add_argument("--resolution", "-r", type=float, default=0.1, help="Grid resolution in meters")
    parser.add_argument("--dimensions", "-d", type=int, default=16000, help="VSA dimensions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Processing options
    parser.add_argument("--incremental", "-i", action="store_true", help="Use incremental processing")
    parser.add_argument("--horizon", type=float, default=10.0, help="Horizon distance for incremental processing")
    parser.add_argument("--sample-resolution", type=float, help="Resolution for sampling grid in incremental processing")
    parser.add_argument("--max-samples", type=int, help="Maximum number of sample positions for incremental processing")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing points")
    parser.add_argument("--length-scale", type=float, default=2.0, help="Length scale for power operation")
    parser.add_argument("--min-cell-resolution", type=float, help="Minimum resolution for spatial indexing")
    parser.add_argument("--max-cell-resolution", type=float, help="Maximum resolution for spatial indexing")
    parser.add_argument("--cache-size", type=int, default=10000, help="Maximum size of vector cache")
    parser.add_argument("--memory-threshold", type=float, default=0.8, help="Threshold for GPU memory usage (0.0-1.0)")
    parser.add_argument("--safety-margin", type=float, default=0.5, help="Minimum distance from occupied points for sampling")
    
    # Enhanced options
    parser.add_argument("--enhanced", "-e", action="store_true", help="Use enhanced mapper")
    parser.add_argument("--save-stats", "-s", action="store_true", help="Save processing statistics")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    
    args = parser.parse_args()
    
    # Call the main function with CLI arguments
    pointcloud_to_ogm(
        args.input,
        args.output,
        world_bounds=args.bounds,
        resolution=args.resolution,
        vsa_dimensions=args.dimensions,
        use_cuda=not args.cpu,
        verbose=args.verbose,
        incremental=args.incremental,
        horizon_distance=args.horizon,
        sample_resolution=args.sample_resolution,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        length_scale=args.length_scale,
        min_cell_resolution=args.min_cell_resolution,
        max_cell_resolution=args.max_cell_resolution,
        cache_size=args.cache_size,
        memory_threshold=args.memory_threshold,
        safety_margin=args.safety_margin,
        use_enhanced_mapper=args.enhanced,
        save_stats=args.save_stats,
        visualize=args.visualize
    )
```

3. **Enhanced Example Script**

```python
def enhanced_example():
    """Run an example using the enhanced VSA mapper."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    output_file = "outputs/enhanced_grid.npz"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    print("\n=== Example: Enhanced VSA Mapper ===\n")
    
    # Convert a point cloud to an occupancy grid map using the enhanced mapper
    result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=output_file,
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        batch_size=1000,
        cache_size=10000,
        memory_threshold=0.8,
        safety_margin=0.5,
        use_enhanced_mapper=True,
        save_stats=True,
        visualize=True,
        verbose=True
    )
    
    # Print statistics
    if "stats" in result:
        stats = result["stats"]
        print("\n=== Enhanced VSA Mapper Statistics ===")
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Process time: {stats['process_time']:.2f} seconds")
        print(f"Incremental time: {stats['incremental_time']:.2f} seconds")
        print(f"Total points processed: {stats['total_points_processed']}")
        print(f"Total samples processed: {stats['total_samples_processed']}")
        print(f"Points per second: {stats['points_per_second']:.2f}")
        print(f"Cache hit rate: {stats['cache_hit_rate']*100:.2f}%")
        
        if "current_memory_gb" in stats:
            print(f"Memory usage: {stats['current_memory_gb']:.2f} GB / {stats['max_memory_gb']:.2f} GB")
    
    return result
```

4. **Performance Comparison Script**

```python
def compare_mappers():
    """Compare the performance of the original and enhanced VSA mappers."""
    # Input and output files
    input_file = "inputs/obstacle_map.npy"
    original_output = "outputs/original_grid.npz"
    enhanced_output = "outputs/enhanced_grid.npz"
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    print("\n=== Comparing Original and Enhanced VSA Mappers ===\n")
    
    # Process with original mapper
    print("\n--- Original VSA Mapper ---\n")
    original_result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=original_output,
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        use_enhanced_mapper=False,
        visualize=True,
        verbose=True
    )
    
    # Process with enhanced mapper
    print("\n--- Enhanced VSA Mapper ---\n")
    enhanced_result = pointcloud_to_ogm(
        input_file=input_file,
        output_file=enhanced_output,
        world_bounds=[-50, 50, -50, 50],
        resolution=0.1,
        incremental=True,
        horizon_distance=10.0,
        sample_resolution=1.0,
        max_samples=100,
        safety_margin=0.5,
        use_enhanced_mapper=True,
        save_stats=True,
        visualize=True,
        verbose=True
    )
    
    # Print performance comparison
    print("\n=== Performance Comparison ===\n")
    print(f"Original processing time: {original_result['processing_time']:.2f} seconds")
    print(f"Enhanced processing time: {enhanced_result['processing_time']:.2f} seconds")
    
    # Calculate speedup/slowdown
    speedup = original_result['processing_time'] / enhanced_result['processing_time']
    if speedup > 1:
        print(f"Enhanced mapper is {speedup:.2f}x faster than original mapper")
    else:
        print(f"Enhanced mapper is {1/speedup:.2f}x slower than original mapper")
    
    # Print enhanced mapper statistics
    if "stats" in enhanced_result:
        stats = enhanced_result["stats"]
        print("\n=== Enhanced VSA Mapper Statistics ===")
        print(f"Process time: {stats['process_time']:.2f} seconds")
        print(f"Incremental time: {stats['incremental_time']:.2f} seconds")
        print(f"Total points processed: {stats['total_points_processed']}")
        print(f"Total samples processed: {stats['total_samples_processed']}")
        print(f"Points per second: {stats['points_per_second']:.2f}")
        print(f"Cache hit rate: {stats['cache_hit_rate']*100:.2f}%")
    
    return original_result, enhanced_result
```

5. **Updated Main Module**

```python
def main():
    """Main entry point when run as a module."""
    if len(sys.argv) > 1:
        # If arguments are provided, run CLI
        cli_main()
    else:
        # If no arguments, run example
        print("No arguments provided. Running example...")
        print("For CLI usage, run with --help flag.")
        
        # Run enhanced example
        enhanced_example()

# Allow running as a script
if __name__ == "__main__":
    main()
```

### Testing Plan

1. **Unit Tests for Main Interface**

```python
def test_main_interface():
    """Test the main interface with both mappers."""
    # Create a small test grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 1, 0, 1]
    resolution = 0.1
    
    # Create temporary files
    input_file = "test_input.npy"
    original_output = "test_original.npz"
    enhanced_output = "test_enhanced.npz"
    
    try:
        # Save test point cloud
        points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
        points_np = points.cpu().numpy()
        labels_np = labels.cpu().numpy()
        combined = np.column_stack((points_np, labels_np))
        np.save(input_file, combined)
        
        # Test with original mapper
        original_result = pointcloud_to_ogm(
            input_file=input_file,
            output_file=original_output,
            world_bounds=world_bounds,
            resolution=resolution,
            vsa_dimensions=1000,  # Small for testing
            use_cuda=False,  # Use CPU for testing
            verbose=False,
            incremental=True,
            horizon_distance=0.5,
            sample_resolution=0.2,
            max_samples=5,
            use_enhanced_mapper=False
        )
        
        # Verify original result
        assert "grid" in original_result
        assert "metadata" in original_result
        assert "processing_time" in original_result
        assert original_result["metadata"]["use_enhanced_mapper"] == False
        
        # Test with enhanced mapper
        enhanced_result = pointcloud_to_ogm(
            input_file=input_file,
            output_file=enhanced_output,
            world_bounds=world_bounds,
            resolution=resolution,
            vsa_dimensions=1000,  # Small for testing
            use_cuda=False,  # Use CPU for testing
            verbose=False,
            incremental=True,
            horizon_distance=0.5,
            sample_resolution=0.2,
            max_samples=5,
            safety_margin=0.1,
            use_enhanced_mapper=True,
            save_stats=True
        )
        
        # Verify enhanced result
        assert "grid" in enhanced_result
        assert "metadata" in enhanced_result
        assert "processing_time" in enhanced_result
        assert "stats" in enhanced_result
        assert enhanced_result["metadata"]["use_enhanced_mapper"] == True
        
        # Load saved files
        original_data = np.load(original_output)
        enhanced_data = np.load(enhanced_output)
        
        # Verify saved data
        assert "grid" in original_data
        assert "grid" in enhanced_data
        assert "use_enhanced_mapper" in enhanced_data
        assert enhanced_data["use_enhanced_mapper"] == True
        
    finally:
        # Clean up temporary files
        for file in [input_file, original_output, enhanced_output]:
            if os.path.exists(file):
                os.remove(file)
```

2. **CLI Tests**

```python
def test_cli():
    """Test the CLI interface."""
    # Create a small test grid
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[4:7, 4:7] = 1  # Add a small square in the middle
    
    # Convert to point cloud
    world_bounds = [0, 1, 0, 1]
    resolution = 0.1
    
    # Create temporary files
    input_file = "test_input.npy"
    output_file = "test_output.npz"
    
    try:
        # Save test point cloud
        points, labels = convert_occupancy_grid_to_pointcloud(grid, world_bounds, resolution)
        points_np = points.cpu().numpy()
        labels_np = labels.cpu().numpy()
        combined = np.column_stack((points_np, labels_np))
        np.save(input_file, combined)
        
        # Test CLI with original mapper
        original_cmd = [
            sys.executable, "-m", "vsa_ogm.main",
            input_file, output_file,
            "--bounds", "0", "1", "0", "1",
            "--resolution", "0.1",
            "--dimensions", "1000",
            "--cpu",
            "--verbose",
            "--incremental",
            "--horizon", "0.5",
            "--sample-resolution", "0.2",
            "--max-samples", "5"
        ]
        
        subprocess.run(original_cmd, check=True)
        
        # Verify output file exists
        assert os.path.exists(output_file)
        
        # Load saved file
        original_data = np.load(output_file)
        
        # Verify saved data
        assert "grid" in original_data
        
        # Remove output file
        os.remove(output_file)
        
        # Test CLI with enhanced mapper
        enhanced_cmd = [
            sys.executable, "-m", "vsa_ogm.main",
            input_file, output_file,
            "--bounds", "0", "1", "0", "1",
            "--resolution", "0.1",
            "--dimensions", "1000",
            "--cpu",
            "--verbose",
            "--incremental",
            "--horizon", "0.5",
            "--sample-resolution", "0.2",
            "--max-samples", "5",
            "--safety-margin", "0.1",
            "--enhanced",
            "--save-stats",
            "--visualize"
        ]
        
        subprocess.run(enhanced_cmd, check=True)
        
        # Verify output files exist
        assert os.path.exists(output_file)
        assert os.path.exists(os.path.splitext(output_file)[0] + "_visualization.png")
        assert os.path.exists(os.path.splitext(output_file)[0] + "_class.png")
        
        # Load saved file
        enhanced_data = np.load(output_file)
        
        # Verify saved data
        assert "grid" in enhanced_data
        assert "use_enhanced_mapper" in enhanced_data
        assert enhanced_data["use_enhanced_mapper"] == True
        
    finally:
        # Clean up temporary files
        for file in [input_file, output_file]:
            if os.path.exists(file):
                os.remove(file)
        
        # Clean up visualization files
        for file in [os.path.splitext(output_file)[0] + "_visualization.png", os.path.splitext(output_file)[0] + "_class.png"]:
            if os.path.exists(file):
                os.remove(file)
```

### Integration with Existing Code

The updated main interface and CLI will be integrated into the existing codebase by:
1. Updating `src/main.py` with the new `pointcloud_to_ogm` function and `cli_main` function
2. Adding the enhanced example and comparison scripts to `examples/enhanced_usage.py`
3. Ensuring backward compatibility with existing code that uses the original interface

### Expected Outcomes

1. **Improved Usability**: The updated interface will make it easy for users to leverage the enhanced VSA mapper and its new features.
2. **Backward Compatibility**: Existing code that uses the original interface will continue to work without changes.
3. **Performance Monitoring**: Users will be able to track and compare the performance of the original and enhanced mappers.
4. **Visualization**: Users will be able to visualize the results of the mapping process.

### Next Steps

After implementing the updated main interface and CLI, we will proceed to Phase 5, which will focus on comprehensive testing and documentation to ensure the enhanced VSA-OGM implementation is robust and user-friendly.
