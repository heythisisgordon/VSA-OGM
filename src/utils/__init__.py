"""
Utility functions for Sequential VSA-OGM.

This package provides utility functions for the Sequential VSA-OGM system,
including visualization, metrics, and I/O operations.
"""

from src.utils.visualization import (
    plot_occupancy_grid,
    plot_entropy_map,
    plot_classification,
    plot_point_cloud,
    plot_quadrants,
    plot_sample_positions,
    plot_combined_results,
    plot_sensor_readings,
    save_figure
)

from src.utils.metrics import (
    calculate_auc,
    plot_auc,
    calculate_precision_recall_f1,
    calculate_confusion_matrix,
    plot_confusion_matrix,
    calculate_iou,
    calculate_accuracy,
    calculate_runtime_metrics,
    plot_runtime_metrics,
    compare_with_ground_truth
)

from src.utils.io import (
    load_point_cloud,
    save_point_cloud,
    load_occupancy_grid,
    save_occupancy_grid,
    load_config,
    save_config,
    load_labels,
    save_labels,
    load_results,
    save_results,
    create_directory,
    list_files,
    load_memory_vectors,
    save_memory_vectors
)

__all__ = [
    # Visualization
    'plot_occupancy_grid',
    'plot_entropy_map',
    'plot_classification',
    'plot_point_cloud',
    'plot_quadrants',
    'plot_sample_positions',
    'plot_combined_results',
    'plot_sensor_readings',
    'save_figure',
    
    # Metrics
    'calculate_auc',
    'plot_auc',
    'calculate_precision_recall_f1',
    'calculate_confusion_matrix',
    'plot_confusion_matrix',
    'calculate_iou',
    'calculate_accuracy',
    'calculate_runtime_metrics',
    'plot_runtime_metrics',
    'compare_with_ground_truth',
    
    # I/O
    'load_point_cloud',
    'save_point_cloud',
    'load_occupancy_grid',
    'save_occupancy_grid',
    'load_config',
    'save_config',
    'load_labels',
    'save_labels',
    'load_results',
    'save_results',
    'create_directory',
    'list_files',
    'load_memory_vectors',
    'save_memory_vectors'
]
