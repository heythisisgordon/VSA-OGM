import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import random
import pickle as pkl
import torch
import yaml

import vsa_ogm.dataloaders.functional as hogmf
from vsa_ogm.metrics import calculate_AUC
from vsa_ogm.utilities import train_test_split
import spl.mapping as spm

torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

BASE_CONFIG: dict = {
    "experiment_name": "build_vsa_map_occupancy_grid",
    "verbose": True,
    "mapper": {
        "axis_resolution": 0.5,  # meters
        "decision_thresholds": [-0.99, 0.99],
        "device": "cpu" if torch.cuda.is_available() else "cpu",
        "length_scale": 2.0,
        "quadrant_hierarchy": [4],
        "use_query_normalization": True,
        "use_query_rescaling": False,
        "verbose": True,
        "vsa_dimensions": 16000,
        "plotting": {
            "plot_xy_voxels": False
        }
    },
    "data": {
        "dataset_name": "occupancy_grid",  # toysim, intel, occupancy_grid
        "test_split": 0.1,
        "occupancy_grid": {
            "data_dir": "datasets/obstacle_map.npy",
            "world_bounds": [-50, 50, -50, 50],  # x_min, x_max, y_min, y_max; meters
            "resolution": 0.1  # Meters per cell
        },
    },
    "logging": {
        "log_dir": "vsa_ogm/experiments/logs",
        "occupied_map_dir": "occupied_maps",
        "empty_map_dir": "empty_maps",
        "global_maps_dir": "global_maps",
        "run_time_metrics": "run_time_metrics.pkl"
    }
}


def main(config: dict) -> None:
    """
    Main function for building a VSA map from an occupancy grid.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        None
    """

    config: DictConfig = DictConfig(config)

    if config.verbose:
        print("--- Building VSA Map from Occupancy Grid ---")
        print(yaml.dump(OmegaConf.to_container(config)))

    dataloader, world_size = hogmf.load_single_data(config)

    log_dir: str = config.logging.log_dir
    log_path: str = os.path.join(log_dir, config.experiment_name)
    config_save_path: str = os.path.join(log_path, "config.yaml")

    if config.verbose:
        print(f"Log Path: {log_path}")
        print(f"Config Save Path: {config_save_path}\n")

    os.makedirs(log_path, exist_ok=True)
    with open(config_save_path, "w") as f:
        f.write(yaml.dump(OmegaConf.to_container(config)))

    output = dataloader.reset()

    mapper = spm.OGM2D_V4(
        config["mapper"],
        world_size,
        log_dir=log_path
    )
    test_split: float = config.data.test_split
    test_data: dict = {
        "lidar_data": [],
        "occupancy": []
    }

    # Since we only have one step for the occupancy grid dataloader,
    # we'll just process it once
    if config.verbose:
        print(f"Processing Occupancy Grid")

    train_lidar, train_occupancy, test_lidar, test_occupancy = train_test_split(output, test_split)
    test_data["lidar_data"].append(test_lidar)
    test_data["occupancy"].append(test_occupancy)

    mapper.process_observation(train_lidar, train_occupancy)

    if config.verbose:
        print("Processing Test Data...")

    test_data_path: str = os.path.join(
        log_dir, "test_data.pkl"
    )
    with open(test_data_path, "wb") as f:
        pkl.dump(test_data, f)

    # calculate_AUC(
    #     mapper=mapper,
    #     test_data=test_data,
    #     log_dir=log_path
    # )


if __name__ == "__main__":
    main(BASE_CONFIG)
