
import os
import yaml
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def load_yaml_file(file_path):
    path = Path(file_path)
    if not path.exists():
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def deep_merge(base, override):
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def load_config(config_path=None, cli_overrides=None):
    project_root = get_project_root()
    
    base_config_path = project_root / 'configs' / 'base.yaml'
    config = load_yaml_file(base_config_path)
    
    if config_path:
        custom_config = load_yaml_file(config_path)
        config = deep_merge(config, custom_config)
    
    if cli_overrides:
        config = deep_merge(config, cli_overrides)
    
    return config

def get_secrets():
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
    path = Path(path_str)
    
    if path.is_absolute():
        return path
    
    return get_project_root() / path

def print_config(config, indent=0):
    prefix = "  " * indent
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
