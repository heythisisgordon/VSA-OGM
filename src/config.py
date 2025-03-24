"""
Configuration management for Sequential VSA-OGM.

This module provides configuration management for the Sequential VSA-OGM system,
including default parameters, validation, and loading/saving configurations.
"""

import os
import json
from typing import Dict, Any, Optional, Union


class Config:
    """
    Configuration manager for Sequential VSA-OGM.
    
    This class manages configuration parameters for the Sequential VSA-OGM system,
    providing default values, validation, and loading/saving functionality.
    """
    
    # Default configuration parameters
    DEFAULT_CONFIG = {
        # Vector Symbolic Architecture parameters
        "vsa": {
            "dimensions": 1024,           # Dimensionality of VSA vectors
            "length_scale": 1.0,          # Length scale for fractional binding
        },
        
        # Quadrant memory parameters
        "quadrant": {
            "size": 8,                    # Number of quadrants along each axis (total = size^2)
        },
        
        # Sequential processing parameters
        "sequential": {
            "sample_resolution": 1.0,     # Distance between sample points on the grid
            "sensor_range": 10.0,         # Maximum range of the simulated sensor
        },
        
        # Entropy extraction parameters
        "entropy": {
            "disk_radius": 3,             # Radius of the disk filter for local entropy calculation
            "occupied_threshold": 0.6,     # Threshold above which a cell is classified as occupied
            "empty_threshold": 0.3,        # Threshold below which a cell is classified as empty
        },
        
        # System parameters
        "system": {
            "device": "cpu",              # Device to perform calculations on ("cpu" or "cuda")
            "show_progress": True,        # Whether to show progress bars
        }
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the configuration manager.
        
        Args:
            config_dict: Optional dictionary with configuration parameters
        """
        # Start with default configuration
        self.config = self._deep_copy_dict(self.DEFAULT_CONFIG)
        
        # Update with provided configuration if any
        if config_dict is not None:
            self._update_config(config_dict)
            
        # Validate configuration
        self._validate_config()
        
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        return json.loads(json.dumps(d))
        
    def _update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with provided dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters to update
        """
        for section, params in config_dict.items():
            if section in self.config:
                if isinstance(params, dict):
                    for key, value in params.items():
                        if key in self.config[section]:
                            self.config[section][key] = value
                        else:
                            raise ValueError(f"Unknown parameter '{key}' in section '{section}'")
                else:
                    raise ValueError(f"Section '{section}' should be a dictionary")
            else:
                raise ValueError(f"Unknown section '{section}'")
                
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # VSA parameters
        assert isinstance(self.config["vsa"]["dimensions"], int), "VSA dimensions must be an integer"
        assert self.config["vsa"]["dimensions"] > 0, "VSA dimensions must be positive"
        assert isinstance(self.config["vsa"]["length_scale"], (int, float)), "Length scale must be a number"
        assert self.config["vsa"]["length_scale"] > 0, "Length scale must be positive"
        
        # Quadrant parameters
        assert isinstance(self.config["quadrant"]["size"], int), "Quadrant size must be an integer"
        assert self.config["quadrant"]["size"] > 0, "Quadrant size must be positive"
        
        # Sequential parameters
        assert isinstance(self.config["sequential"]["sample_resolution"], (int, float)), "Sample resolution must be a number"
        assert self.config["sequential"]["sample_resolution"] > 0, "Sample resolution must be positive"
        assert isinstance(self.config["sequential"]["sensor_range"], (int, float)), "Sensor range must be a number"
        assert self.config["sequential"]["sensor_range"] > 0, "Sensor range must be positive"
        
        # Entropy parameters
        assert isinstance(self.config["entropy"]["disk_radius"], int), "Disk radius must be an integer"
        assert self.config["entropy"]["disk_radius"] > 0, "Disk radius must be positive"
        assert isinstance(self.config["entropy"]["occupied_threshold"], (int, float)), "Occupied threshold must be a number"
        assert 0 <= self.config["entropy"]["occupied_threshold"] <= 1, "Occupied threshold must be between 0 and 1"
        assert isinstance(self.config["entropy"]["empty_threshold"], (int, float)), "Empty threshold must be a number"
        assert 0 <= self.config["entropy"]["empty_threshold"] <= 1, "Empty threshold must be between 0 and 1"
        
        # System parameters
        assert self.config["system"]["device"] in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'"
        assert isinstance(self.config["system"]["show_progress"], bool), "Show progress must be a boolean"
        
    def get(self, section: str, param: Optional[str] = None) -> Any:
        """
        Get configuration parameter(s).
        
        Args:
            section: Configuration section
            param: Optional parameter name within section
            
        Returns:
            Configuration parameter value or section dictionary
        """
        if section not in self.config:
            raise ValueError(f"Unknown section '{section}'")
            
        if param is None:
            return self.config[section]
        
        if param not in self.config[section]:
            raise ValueError(f"Unknown parameter '{param}' in section '{section}'")
            
        return self.config[section][param]
    
    def set(self, section: str, param: str, value: Any) -> None:
        """
        Set configuration parameter.
        
        Args:
            section: Configuration section
            param: Parameter name within section
            value: Parameter value
        """
        if section not in self.config:
            raise ValueError(f"Unknown section '{section}'")
            
        if param not in self.config[section]:
            raise ValueError(f"Unknown parameter '{param}' in section '{section}'")
            
        self.config[section][param] = value
        
        # Validate configuration after update
        self._validate_config()
        
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration file
        """
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Config object with loaded configuration
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file '{filepath}' not found")
            
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with configuration parameters
        """
        return self._deep_copy_dict(self.config)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.config, indent=2)
