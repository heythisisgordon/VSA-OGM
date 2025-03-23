# Refactor 25: Remove Standard Processing Capabilities and Rename Incremental to Sequential

## Overview

This refactoring plan outlines the steps to completely remove all "Standard Processing" capabilities from the VSA-OGM codebase and rename "Incremental Processing" to "Sequential Processing". Standard Processing refers to processing an entire point cloud at once, which is inefficient and should be replaced entirely with sequential processing.

## Motivation

Processing entire point clouds at once is:
- Memory inefficient for large datasets
- Computationally expensive
- Not necessary given the sequential processing capabilities

Sequential processing (formerly "incremental") is superior because it:
- Processes point clouds in smaller chunks based on spatial locality
- Reduces memory usage
- Can be more efficient for large point clouds
- Provides the same quality results

## Implementation Plan

### Phase 1: Rename Incremental to Sequential

#### 1. Rename in `src/mapper.py`

- Rename the method `process_incrementally` to `process_sequentially`
- Update all internal variable names and comments that reference "incremental" to "sequential"
- Update docstrings to reflect the new terminology

#### 2. Rename in `src/main.py`

- Rename the parameter `incremental` to `sequential` (before removing it in Phase 2)
- Rename any variables like `incremental_processing` to `sequential_processing`
- Update the command-line interface to change `--incremental` to `--sequential` (before removing it in Phase 2)
- Update help text and documentation strings

#### 3. Rename in `src/functional.py`

- Rename any functions or variables with "incremental" in their name to use "sequential" instead
- Update function documentation to use the new terminology

#### 4. Rename in `src/utils.py`

- Update any utility functions or variables that reference "incremental" to use "sequential"
- Update visualization function parameters and documentation

### Phase 2: Remove Standard Processing

#### 1. Modify `src/mapper.py`

- Remove the `process_observation` method that processes the entire point cloud at once
- Modify the `__init__` method to remove any parameters or initialization related to standard processing
- Update the `VSAMapper` class to only support sequential processing
- Ensure all internal methods assume sequential processing

#### 2. Modify `src/main.py`

- Remove the `sequential` parameter from the `pointcloud_to_ogm` function, as sequential will be the only option
- Update the function to always use sequential processing
- Remove any conditional logic that checks for the `sequential` parameter
- Update the command-line interface to remove the `--sequential` flag (since it will be the default behavior)

#### 3. Update `src/functional.py`

- Remove any functions that are only used for standard processing
- Update functions that are used by both standard and sequential processing to only support the sequential workflow

#### 4. Update `src/utils.py`

- Remove any utility functions that are specific to standard processing
- Update visualization functions to remove references to standard processing

### Phase 3: Test Updates

#### 1. Modify `tests/test_vsa.py`

- Rename `test_incremental_processing` to `test_sequential_processing` if it exists
- Remove `test_performance_comparison` which compares standard and sequential processing
- Update `test_vsa_mapper` to only test sequential processing
- Remove any other tests that rely on standard processing
- Update `run_all_tests` to remove references to removed tests

#### 2. Modify `tests/test_integration.py`

- Update `test_full_pipeline` to only use sequential processing
- Rename `test_incremental_parameters` to `test_sequential_parameters` and ensure it doesn't imply a comparison with standard processing
- Ensure all tests assume sequential processing

#### 3. Modify `tests/benchmark.py`

- Remove `benchmark_processing_modes` which compares standard and sequential processing
- Rename functions like `benchmark_incremental_parameters` to `benchmark_sequential_parameters`
- Update other benchmark functions to only test sequential processing with different parameters
- Update `run_all_benchmarks` to remove references to removed benchmarks

### Phase 4: Documentation Updates

#### 1. Update `README.md`

- Remove references to standard processing
- Rename "incremental processing" to "sequential processing" throughout
- Update examples to only show sequential processing
- Remove the "Advanced Features" section that contrasts standard and sequential processing
- Update command-line interface examples to remove the `--sequential` flag

#### 2. Update `docs/api.md`

- Remove references to standard processing in the API documentation
- Rename all instances of "incremental" to "sequential"
- Update the `pointcloud_to_ogm` function documentation to remove the `sequential` parameter
- Update the `VSAMapper` class documentation to remove the `process_observation` method
- Update other method documentation to reflect the sequential-only approach

#### 3. Update `docs/examples.md`

- Remove examples that use standard processing
- Rename all instances of "incremental" to "sequential"
- Update all examples to only show sequential processing
- Remove comparisons between standard and sequential processing

#### 4. Update `docs/performance.md`

- Remove sections that compare standard and sequential processing
- Rename all instances of "incremental" to "sequential"
- Update performance guidelines to focus on optimizing sequential processing
- Remove references to standard processing in performance tips

### Phase 5: Final Cleanup

#### 1. Code Review

- Perform a comprehensive code review to ensure all references to standard processing have been removed
- Check for any lingering comments, variable names, or function parameters that reference standard processing
- Ensure all "incremental" terminology has been replaced with "sequential"

#### 2. Documentation Review

- Review all documentation to ensure consistency with the sequential-only approach
- Ensure all "incremental" terminology has been replaced with "sequential"
- Update any diagrams or visualizations that show standard processing or use "incremental" terminology

#### 3. Testing

- Run all tests to ensure they pass with the sequential-only implementation
- Verify that all functionality works as expected without standard processing

## Expected Outcomes

After implementing this refactoring plan:

1. The VSA-OGM codebase will only support sequential processing (formerly "incremental")
2. All documentation will reflect the sequential-only approach
3. Tests will only test sequential processing with different parameters
4. The API will be simpler and more focused
5. Memory usage will be more efficient for large point clouds
6. Terminology will be more consistent and clearer with "sequential" instead of "incremental"

## Timeline

This refactoring should be completed in a single phase to ensure consistency across the codebase. Estimated time: 1-2 days.
