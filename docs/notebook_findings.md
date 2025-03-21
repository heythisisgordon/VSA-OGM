# Important Findings from Notebooks

This document summarizes the important findings and insights from the original Jupyter notebooks that were used during the development of VSA-OGM.

## Ablation Study

The `ablation.ipynb` notebook contains experiments that analyze the impact of different parameters on the performance of VSA-OGM. Key findings include:

- The impact of VSA dimensionality on mapping accuracy
- The effect of length scale on the resolution of the occupancy grid
- The trade-off between computational efficiency and mapping accuracy

## Entropy Information Extraction

The `entropy_information_extraction.ipynb` notebook explores the use of Shannon entropy for information extraction from the occupancy grid. Key findings include:

- How entropy can be used to identify areas of uncertainty in the map
- The relationship between entropy and information content
- Methods for extracting useful information from the entropy map

## Fusion Experimentation

The `fusion_experimentation.ipynb` and `intel-fusion-experimentation.ipynb` notebooks contain experiments on fusing multiple observations into a single occupancy grid. Key findings include:

- Methods for combining observations from multiple agents
- The impact of different fusion strategies on mapping accuracy
- The benefits of multi-agent mapping compared to single-agent mapping

## Runtime Analysis

The `runtime_plotting.ipynb` notebook contains analysis of the runtime performance of VSA-OGM. Key findings include:

- The computational complexity of different parts of the algorithm
- The impact of different parameters on runtime
- Strategies for improving computational efficiency

## Test Data Generation

The `test_data_generator.ipynb` notebook contains code for generating test data for VSA-OGM. Key findings include:

- Methods for generating synthetic point clouds
- Strategies for creating challenging test cases
- Approaches for evaluating the performance of VSA-OGM on different types of data

## Datasets

The original notebooks contained code for working with different datasets:

- Working with the EviLOG dataset
- Generating fusion data from the Intel map
- Working with simulation data

These notebooks provided insights into how to work with different types of data and how to prepare data for use with VSA-OGM.
