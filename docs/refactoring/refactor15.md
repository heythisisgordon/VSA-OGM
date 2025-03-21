# Cleanup Plan for VSA-OGM Refactoring

## Overview

This document outlines a plan to clean up the VSA-OGM project after the initial refactoring to a simplified structure. The goal is to remove directories that are not part of the simplified structure and ensure that any necessary code is properly migrated.

## Directories to Remove

1. `/assets` - Contains GIF files used for documentation
2. `/build` - Build artifacts
3. `/datasets` - Data files
4. `/notebooks` - Jupyter notebooks from original development
5. `/spl` - Original SPL package
6. `/vsa_ogm` - Original VSA-OGM package

## Assessment and Migration Plan

### 1. /assets

**Assessment:**
- Contains GIF files (toy-sim.gif, vsa-toysim-crop.gif)
- These may be used for documentation or examples

**Migration Plan:**
1. Check if these files are referenced in the README.md or other documentation
2. If needed, move them to a new `/docs/assets` directory
3. Update any references to point to the new location

### 2. /build

**Assessment:**
- Contains build artifacts
- These are generated files that can be safely deleted

**Migration Plan:**
1. Delete the directory
2. Add `/build` to .gitignore to prevent it from being committed in the future

### 3. /datasets

**Assessment:**
- Contains data files used for testing and examples
- These may be referenced in the code

**Migration Plan:**
1. Check if these files are referenced in the code
2. If needed, move essential test data to a new `/tests/data` directory
3. Move example data to `/examples/data`
4. Update any references in the code to point to the new locations

### 4. /notebooks

**Assessment:**
- Contains Jupyter notebooks from the original development
- These are not part of the simplified structure

**Migration Plan:**
1. Extract any essential code or algorithms that haven't been migrated to the src directory
2. Document any important findings or insights in the appropriate documentation
3. Delete the directory

### 5. /spl

**Assessment:**
- Contains the original SPL package
- The refactoring plan mentions that the core VSA-OGM algorithm is in this package

**Migration Plan:**
1. Check if all essential code has been migrated to the src directory
2. Specifically, verify that the following have been migrated:
   - `spl/mapping.py` -> `src/mapper.py`
   - `spl/functional.py` -> `src/functional.py`
3. If any code is missing, migrate it to the appropriate files in the src directory
4. Delete the directory once all essential code has been migrated

### 6. /vsa_ogm

**Assessment:**
- Contains the original VSA-OGM package
- This is being replaced by the src directory

**Migration Plan:**
1. Check if all essential code has been migrated to the src directory
2. Specifically, verify that the following have been migrated:
   - `vsa_ogm/main.py` -> `src/main.py`
   - `vsa_ogm/mapper.py` -> `src/mapper.py`
   - `vsa_ogm/functional.py` -> `src/functional.py`
   - `vsa_ogm/io.py` -> `src/io.py`
   - `vsa_ogm/utilities.py` -> `src/utils.py`
   - Any essential code from `vsa_ogm/dataloaders/` and `vsa_ogm/experiments/`
3. If any code is missing, migrate it to the appropriate files in the src directory
4. Delete the directory once all essential code has been migrated

## Implementation Steps

1. **Backup**: Create a backup of the entire project before making any deletions
2. **Verify Migration**: Ensure all essential code has been migrated to the src directory
3. **Run Tests**: Run all tests to ensure the refactored code works correctly
4. **Remove Directories**: Remove the directories in the following order:
   - /build (safe to delete immediately)
   - /notebooks (after extracting any essential code)
   - /spl (after verifying all essential code has been migrated)
   - /vsa_ogm (after verifying all essential code has been migrated)
   - /assets (after moving any needed files to /docs/assets)
   - /datasets (after moving any needed files to /tests/data and /examples/data)
5. **Update Documentation**: Update any documentation that references the removed directories
6. **Final Testing**: Run all tests again to ensure everything still works correctly

## Expected Benefits

- Cleaner project structure
- Reduced confusion with duplicate code
- Smaller repository size
- Easier to understand and maintain
