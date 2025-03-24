Specific inefficiencies in our implementation compared to the original VSA-OGM:

1. **Vector dimensionality issues** - The original code in `spl/mapping.py` uses adaptive dimensionality (configurable via `config.vsa_dimensions`). Your implementation in `src/config.py` defaults to only 1024 dimensions, which is much smaller than what the original uses for complex environments (up to 200,000 dimensions in some examples).

2. **Quadrant memory architecture** - The original implementation in `spl/mapping.py` uses a direct mapping approach with tensor operations for quadrant assignment, while your implementation in `src/quadrant_memory.py` adds an extra layer of dictionary lookup with `self.quadrant_indices` that creates Python overhead.

3. **Sequential processing inefficiency** - Your `src/sequential_processor.py` processes each sample position individually, while the original `spl/mapping.py` uses vectorized operations with direct tensor assignment.

4. **Fractional binding implementation** - The original code in `spl/functional.py` performs fractional binding with direct FFT operations, while your implementation in `src/vector_ops.py` includes additional validation checks that slow down the core operations.

5. **Shannon entropy calculation** - The original implementation uses a GPU-optimized approach to entropy calculation, while your `src/entropy.py` implementation has additional overhead with the `_apply_disk_filter` method.

6. **Memory normalization** - The original code in `spl/mapping.py` only normalizes memory vectors after entire batches, while your `src/quadrant_memory.py` normalizes more frequently in `normalize_memories()`.

7. **Redundant point encoding** - Your implementation encodes points for each query, while the original pre-computes and stores encoded points in the `_build_xy_axis_matrix()` method.

8. **Grid resolution issues** - In your `src/config.py`, the default `sample_resolution` is set to 1.0, which is significantly coarser than what's used in the original examples (often 0.1-0.2).

9. **Excessive function calling** - Your implementation splits operations across more classes and methods (`VSAMapper`, `QuadrantMemory`, `SequentialProcessor`, etc.), introducing function call overhead compared to the more integrated approach in the original.

10. **Logging and visualization overhead** - Your implementation in `src/utils/visualization.py` creates more plots and has more monitoring code than the original, which can significantly slow down processing if called frequently.

These inefficiencies combine to create a much slower implementation despite having similar functional components. The original implementation is highly optimized for batch processing and GPU acceleration, which makes a dramatic difference in processing speed.