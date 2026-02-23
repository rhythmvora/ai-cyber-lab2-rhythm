"""
Utility functions
"""

import os
import json


def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Created directory: {directory}")


def load_config(filepath='config.json'):
    """
    Load configuration from JSON file
    
    Args:
        filepath: Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def save_config(config, filepath='config.json'):
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        filepath: Path to save config
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuration saved to: {filepath}")
