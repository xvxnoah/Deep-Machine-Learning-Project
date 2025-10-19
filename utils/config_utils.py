import yaml
import os
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_env():
    """Load environment variables from .env file."""
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("Warning: .env file not found. W&B API key should be set manually.")
    
    return {
        'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
        'WANDB_ENTITY': os.getenv('WANDB_ENTITY'),
        'PROJECT_NAME': os.getenv('PROJECT_NAME', 'Deep Machine Learning Project')
    }


def merge_configs(base_config, override_config):
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_device(device_str='auto'):
    import torch
    
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")
    
    return device


def print_config(config, indent=0):
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f"{key}: {value}")


def validate_config(config):
    required_sections = ['model', 'data', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model config
    if 'num_classes' not in config['model']:
        raise ValueError("model.num_classes must be specified")
    
    # Validate data config
    if 'data_root' not in config['data']:
        raise ValueError("data.data_root must be specified")
    
    # Validate training config
    if 'epochs' not in config['training']:
        raise ValueError("training.epochs must be specified")
    
    print("✓ Configuration validated successfully")
    return True

