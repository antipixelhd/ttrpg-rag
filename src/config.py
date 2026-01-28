# =============================================================================
# Configuration Loading and Merging
# =============================================================================
# This module handles loading YAML config files and merging them with
# command-line overrides. It keeps things simple using plain dictionaries.

import os
import yaml
from pathlib import Path


def get_project_root():
    """
    Get the root directory of the project.
    This is the folder containing main.py and the configs/ directory.
    
    Returns:
        Path: The project root directory
    """
    # Go up from src/ to the project root
    return Path(__file__).parent.parent


def load_yaml_file(file_path):
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        dict: The parsed YAML contents, or empty dict if file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base, override):
    """
    Recursively merge two dictionaries.
    Values in 'override' take precedence over values in 'base'.
    
    Args:
        base: The base dictionary (default values)
        override: The override dictionary (custom values)
        
    Returns:
        dict: A new dictionary with merged values
        
    Example:
        base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        override = {'b': {'x': 99}}
        result = {'a': 1, 'b': {'x': 99, 'y': 20}}
    """
    result = base.copy()
    
    for key, value in override.items():
        # If both are dicts, merge recursively
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            # Otherwise, override the value
            result[key] = value
    
    return result


def load_config(config_path=None, cli_overrides=None):
    """
    Load configuration from YAML files and merge with CLI overrides.
    
    The loading order is:
    1. configs/base.yaml (default values)
    2. Custom config file (if provided via --config)
    3. CLI overrides (highest priority)
    
    Args:
        config_path: Optional path to a custom config YAML file
        cli_overrides: Optional dict of CLI argument overrides
        
    Returns:
        dict: The merged configuration dictionary
    """
    project_root = get_project_root()
    
    # Step 1: Load base config (defaults)
    base_config_path = project_root / 'configs' / 'base.yaml'
    config = load_yaml_file(base_config_path)
    
    # Step 2: Merge custom config file (if provided)
    if config_path:
        custom_config = load_yaml_file(config_path)
        config = deep_merge(config, custom_config)
    
    # Step 3: Apply CLI overrides (if provided)
    if cli_overrides:
        config = deep_merge(config, cli_overrides)
    
    return config


def get_secrets():
    """
    Load API keys and other secrets from configs/secrets.yaml.
    
    Returns:
        dict: Dictionary containing secrets (e.g., openai_api_key)
        
    Raises:
        FileNotFoundError: If secrets.yaml doesn't exist
    """
    project_root = get_project_root()
    secrets_path = project_root / 'configs' / 'secrets.yaml'
    
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Secrets file not found at {secrets_path}\n"
            f"Please copy configs/secrets.example.yaml to configs/secrets.yaml "
            f"and add your API keys."
        )
    
    return load_yaml_file(secrets_path)


def resolve_path(path_str):
    """
    Convert a relative path string to an absolute Path object.
    Relative paths are resolved from the project root.
    
    Args:
        path_str: A path string (can be relative or absolute)
        
    Returns:
        Path: An absolute Path object
    """
    path = Path(path_str)
    
    # If it's already absolute, return as-is
    if path.is_absolute():
        return path
    
    # Otherwise, resolve relative to project root
    return get_project_root() / path


# =============================================================================
# Helper function to print config for debugging
# =============================================================================
def print_config(config, indent=0):
    """
    Pretty-print a configuration dictionary.
    Useful for debugging and logging.
    
    Args:
        config: The configuration dictionary to print
        indent: Current indentation level (used internally for recursion)
    """
    prefix = "  " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
