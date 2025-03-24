"""
Performance and accuracy metrics for Sequential VSA-OGM.

This module provides functions for calculating performance and accuracy metrics
for the Sequential VSA-OGM system, including Area Under Curve (AUC), precision,
recall, and F1 score.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, Optional, List
from sklearn import metrics


def calculate_auc(y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor],
                 threshold_range: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate Area Under the ROC Curve (AUC).
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted scores or probabilities
        threshold_range: Optional array of thresholds to evaluate
        
    Returns:
        Tuple of (AUC score, false positive rates, true positive rates)
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate ROC curve
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    
    # Calculate AUC
    auc_score = metrics.auc(fpr, tpr)
    
    return auc_score, fpr, tpr


def plot_auc(fpr: np.ndarray, 
            tpr: np.ndarray, 
            auc_score: float,
            ax: Optional[plt.Axes] = None,
            title: str = 'ROC Curve',
            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve and AUC.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Plot ROC curve
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc_score:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    
    # Set limits and add legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right')
    
    return fig


def calculate_precision_recall_f1(y_true: Union[np.ndarray, torch.Tensor],
                                 y_pred: Union[np.ndarray, torch.Tensor],
                                 threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted scores or probabilities
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    precision = metrics.precision_score(y_true, y_pred_binary, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred_binary, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_confusion_matrix(y_true: Union[np.ndarray, torch.Tensor],
                              y_pred: Union[np.ndarray, torch.Tensor],
                              threshold: float = 0.5) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted scores or probabilities
        threshold: Threshold for binary classification
        
    Returns:
        Confusion matrix as numpy array
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred_binary)
    
    return cm


def plot_confusion_matrix(cm: np.ndarray,
                         ax: Optional[plt.Axes] = None,
                         title: str = 'Confusion Matrix',
                         figsize: Tuple[int, int] = (8, 6),
                         cmap: str = 'Blues') -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        figsize: Figure size
        cmap: Colormap to use
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    # Set ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    return fig


def calculate_iou(y_true: Union[np.ndarray, torch.Tensor],
                 y_pred: Union[np.ndarray, torch.Tensor],
                 threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted scores or probabilities
        threshold: Threshold for binary classification
        
    Returns:
        IoU score
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate intersection and union
    intersection = np.logical_and(y_true, y_pred_binary).sum()
    union = np.logical_or(y_true, y_pred_binary).sum()
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou


def calculate_accuracy(y_true: Union[np.ndarray, torch.Tensor],
                      y_pred: Union[np.ndarray, torch.Tensor],
                      threshold: float = 0.5) -> float:
    """
    Calculate accuracy.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted scores or probabilities
        threshold: Threshold for binary classification
        
    Returns:
        Accuracy score
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate accuracy
    accuracy = (y_true == y_pred_binary).mean()
    
    return accuracy


def calculate_runtime_metrics(runtimes: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate runtime performance metrics.
    
    Args:
        runtimes: Dictionary with runtime measurements for different components
        
    Returns:
        Dictionary with runtime metrics (mean, std, min, max) for each component
    """
    metrics_dict = {}
    
    for component, times in runtimes.items():
        times_array = np.array(times)
        
        metrics_dict[component] = {
            'mean': np.mean(times_array),
            'std': np.std(times_array),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'total': np.sum(times_array)
        }
        
    return metrics_dict


def plot_runtime_metrics(runtimes: Dict[str, List[float]],
                        ax: Optional[plt.Axes] = None,
                        title: str = 'Runtime Performance',
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot runtime performance metrics.
    
    Args:
        runtimes: Dictionary with runtime measurements for different components
        ax: Optional matplotlib axes to plot on
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    # Calculate metrics
    metrics_dict = calculate_runtime_metrics(runtimes)
    
    # Extract component names and mean runtimes
    components = list(metrics_dict.keys())
    means = [metrics_dict[comp]['mean'] for comp in components]
    stds = [metrics_dict[comp]['std'] for comp in components]
    
    # Create bar plot
    bars = ax.bar(components, means, yerr=stds, capsize=10)
    
    # Add values on top of bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{mean:.3f}s',
               ha='center', va='bottom', rotation=0)
    
    # Set labels and title
    ax.set_xlabel('Component')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def compare_with_ground_truth(prediction: Union[np.ndarray, torch.Tensor],
                             ground_truth: Union[np.ndarray, torch.Tensor],
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    Compare prediction with ground truth and calculate various metrics.
    
    Args:
        prediction: Predicted occupancy grid or probabilities
        ground_truth: Ground truth occupancy grid
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with various metrics
    """
    # Convert to numpy if needed
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
        
    # Flatten arrays
    prediction_flat = prediction.flatten()
    ground_truth_flat = ground_truth.flatten()
    
    # Calculate metrics
    auc_score, _, _ = calculate_auc(ground_truth_flat, prediction_flat)
    pr_metrics = calculate_precision_recall_f1(ground_truth_flat, prediction_flat, threshold)
    iou = calculate_iou(ground_truth_flat, prediction_flat, threshold)
    accuracy = calculate_accuracy(ground_truth_flat, prediction_flat, threshold)
    
    # Combine metrics
    metrics_dict = {
        'auc': auc_score,
        'precision': pr_metrics['precision'],
        'recall': pr_metrics['recall'],
        'f1': pr_metrics['f1'],
        'iou': iou,
        'accuracy': accuracy
    }
    
    return metrics_dict
