"""Base configuration class for DNA-Flex."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar('T', bound='BaseConfig')

@dataclass
class BaseConfig:
    """Base configuration class that all configs should inherit from."""
    
    @classmethod
    def create(cls: Type[T], **kwargs) -> T:
        """Create a config instance with default values overridden by kwargs."""
        return cls(**kwargs)
    
    @classmethod
    def autocreate(cls: Type[T], **kwargs) -> T:
        """Create a config with automatic parameter resolution."""
        return cls.create(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """Update config with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Config has no attribute '{key}'")