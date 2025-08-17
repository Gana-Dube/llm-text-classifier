import yaml
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration loading and validation for the classifier."""
    
    def __init__(self, config_path: str = "config/classifier_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_sections = ['classifier', 'classification_tasks', 'api', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate classifier config
        classifier_config = self.config['classifier']
        required_classifier_keys = ['model_provider', 'model_name', 'temperature', 'max_tokens']
        
        for key in required_classifier_keys:
            if key not in classifier_config:
                raise ValueError(f"Missing required classifier configuration: {key}")
        
        # Validate classification tasks
        tasks = self.config['classification_tasks']
        if not tasks:
            raise ValueError("No classification tasks defined")
        
        for task_name, task_config in tasks.items():
            required_task_keys = ['name', 'description', 'classes', 'prompt_template']
            for key in required_task_keys:
                if key not in task_config:
                    raise ValueError(f"Missing required task configuration '{key}' in task '{task_name}'")
        
        logger.info("Configuration validation passed")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration."""
        return self.config
    
    def get_classifier_config(self) -> Dict[str, Any]:
        """Get classifier-specific configuration."""
        return self.config['classifier']
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for a specific classification task."""
        if task_name not in self.config['classification_tasks']:
            raise ValueError(f"Unknown classification task: {task_name}")
        return self.config['classification_tasks'][task_name]
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available classification tasks."""
        return list(self.config['classification_tasks'].keys())
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config['api']
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config['logging']
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        self._validate_config()
        logger.info("Configuration reloaded successfully")
