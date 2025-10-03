"""
Configuration management for Gemini Invoice Extractor

This module handles loading and accessing configuration from YAML files
with support for environment variable substitution.
"""
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from loguru import logger


class Config:
    """
    Configuration manager for Gemini Invoice Extractor
    
    Loads configuration from YAML file and provides easy access
    to settings with dot notation. Supports environment variable
    substitution for sensitive data like API keys.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file. If None, uses default path.
        """
        if config_path is None:
            # Default to config/gemini_config.yaml
            config_path = Path(__file__).parent.parent.parent / "config" / "gemini_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_env_vars()
        
        logger.info(f"✅ Configuration loaded from: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration file
        
        Returns:
            Dictionary containing configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _resolve_env_vars(self) -> None:
        """
        Recursively resolve environment variables in config
        
        Replaces ${VAR_NAME} with the value of environment variable VAR_NAME
        """
        def resolve_value(value):
            """Resolve environment variables in a value"""
            if isinstance(value, str):
                # Pattern to match ${VAR_NAME}
                pattern = r'\$\{([^}]+)\}'
                matches = re.findall(pattern, value)
                
                for var_name in matches:
                    env_value = os.getenv(var_name)
                    if env_value is not None:
                        value = value.replace(f"${{{var_name}}}", env_value)
                    else:
                        logger.warning(f"⚠️ Environment variable not found: {var_name}")
                
                return value
            
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            
            else:
                return value
        
        self.config = resolve_value(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'api.model_version')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            >>> config.get('api.model_version')
            'gemini-1.5-flash'
            >>> config.get('logging.level')
            'INFO'
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'api.model_version')
            value: Value to set
            
        Examples:
            >>> config.set('logging.level', 'DEBUG')
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section
        
        Args:
            section: Section name (e.g., 'api', 'logging')
            
        Returns:
            Dictionary containing section configuration
            
        Examples:
            >>> config.get_section('api')
            {'api_key': '...', 'model_version': '...', ...}
        """
        return self.get(section, {})
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self.config = self._load_config()
        self._resolve_env_vars()
        logger.info("🔄 Configuration reloaded")
    
    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file
        
        Args:
            output_path: Path to save config. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"💾 Configuration saved to: {output_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Config(path='{self.config_path}')"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get full configuration as dictionary
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid, False otherwise
        """
        required_sections = ['api', 'prompts', 'preprocessing', 'validation']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"❌ Missing required section: {section}")
                return False
        
        # Validate API key is set
        api_key = self.get('api.api_key')
        if not api_key or api_key.startswith('${'):
            logger.error("❌ API key not set. Please configure GEMINI_API_KEY environment variable")
            return False
        
        logger.info("✅ Configuration validation passed")
        return True


# Convenience function to load default config
def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file. If None, uses default.
        
    Returns:
        Config object
        
    Examples:
        >>> from gemini_extractor.config import load_config
        >>> config = load_config()
        >>> model = config.get('api.model_version')
    """
    return Config(config_path)


# Create a global config instance (lazy-loaded)
_global_config: Optional[Config] = None

def get_config() -> Config:
    """
    Get global configuration instance
    
    Returns:
        Global Config object (singleton)
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


if __name__ == "__main__":
    """Test configuration loading"""
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Test config loading
    print("Testing Config class...")
    print("=" * 60)
    
    try:
        config = load_config()
        
        # Test get with dot notation
        print(f"\nAPI Model: {config.get('api.model_version')}")
        print(f"Log Level: {config.get('logging.level')}")
        print(f"Max Image Size: {config.get('preprocessing.max_image_size')}")
        print(f"Strict Mode: {config.get('validation.strict_mode')}")
        
        # Test section retrieval
        print(f"\nAPI Config: {config.get_section('api')}")
        
        # Test validation
        print(f"\nValidation: {config.validate()}")
        
        # Test dictionary access
        print(f"\nDict access - Temperature: {config['api.generation.temperature']}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
