# Reference Guide to Original VSA-OGM Implementation

When developing the Sequential VSA-OGM system, you should focus on understanding and adapting the following key files from the original VSA-OGM implementation contained in the /original_program folder:

## Core Implementation Files

1. **`spl/spl/mapping.py`**
   - Contains the main `OGM2D_V4` class that implements the quadrant-based mapping
   - Study the quadrant hierarchy building and memory vector organization
   - Reference the Shannon entropy implementation and feature extraction

2. **`spl/spl/functional.py`**
   - Contains core VSA operations (bind, power, invert)
   - Implements the mathematical foundation for fractional binding
   - Includes the `make_good_unitary` function for creating basis vectors

3. **`spl/spl/encoders.py`**
   - Provides functions for encoding points in cartesian space
   - Shows how to apply fractional binding to spatial coordinates

4. **`spl/spl/generators.py`**
   - Contains the `SSPGenerator` class for creating axis vectors
   - Important for initializing the VSA system

## Support Files

5. **`vsa_ogm/metrics.py`**
   - Includes functions for calculating AUC and other evaluation metrics
   - Shows how to evaluate mapping performance

6. **`spl/spl/plotting.py`**
   - Contains visualization functions for heatmaps and quadrants
   - Useful for debugging and demonstrating results

7. **`vsa_ogm/utilities.py`**
   - Includes helper functions and utilities

The most important aspect to understand is how `OGM2D_V4` in `mapping.py` manages quadrant memories and applies Shannon entropy for feature extraction. Pay special attention to:

1. The quadrant hierarchy construction (`_build_quadrant_hierarchy`)
2. The point processing approach (`process_observation`)
3. The entropy calculation methods
4. How quadrant memories are updated and normalized

These elements form the foundation of the system's efficiency and will be critical to adapt for the sequential processing approach in your implementation.