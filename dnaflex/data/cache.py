"""Caching system for DNA-Flex data."""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import time
import hashlib
import pickle
from datetime import datetime, timedelta

class DataCache:
    """Cache handler for DNA-Flex data."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_age: int = 86400):
        """Initialize cache handler.
        
        Args:
            cache_dir: Directory for cache storage
            max_age: Maximum age of cached items in seconds (default: 24 hours)
        """
        self.cache_dir = cache_dir or Path.home() / '.dnaflex' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item if found and valid, None otherwise
        """
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                metadata = pickle.load(f)
                data = pickle.load(f)
                
            # Check if cache is expired
            if time.time() - metadata['timestamp'] > self.max_age:
                self.invalidate(key)
                return None
                
            return data
        except (pickle.PickleError, KeyError):
            return None
            
    def set(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """Store item in cache.
        
        Args:
            key: Cache key
            value: Data to cache
            metadata: Optional metadata to store with cached item
        """
        cache_file = self._get_cache_path(key)
        
        metadata = metadata or {}
        metadata.update({
            'timestamp': time.time(),
            'created': datetime.now().isoformat()
        })
        
        with open(cache_file, 'wb') as f:
            pickle.dump(metadata, f)
            pickle.dump(value, f)
            
    def invalidate(self, key: str) -> None:
        """Remove item from cache.
        
        Args:
            key: Cache key
        """
        cache_file = self._get_cache_path(key)
        if cache_file.exists():
            cache_file.unlink()
            
    def clear(self) -> None:
        """Clear all cached items."""
        for cache_file in self.cache_dir.glob('*.cache'):
            cache_file.unlink()
            
    def cleanup(self) -> None:
        """Remove expired cache items."""
        now = time.time()
        for cache_file in self.cache_dir.glob('*.cache'):
            try:
                with open(cache_file, 'rb') as f:
                    metadata = pickle.load(f)
                    
                if now - metadata['timestamp'] > self.max_age:
                    cache_file.unlink()
            except (pickle.PickleError, KeyError):
                # Remove corrupted cache files
                cache_file.unlink()
                
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        # Create deterministic filename from key
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{key_hash}.cache"
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_items': 0,
            'total_size': 0,
            'expired_items': 0,
            'corrupted_items': 0
        }
        
        now = time.time()
        
        for cache_file in self.cache_dir.glob('*.cache'):
            stats['total_items'] += 1
            stats['total_size'] += cache_file.stat().st_size
            
            try:
                with open(cache_file, 'rb') as f:
                    metadata = pickle.load(f)
                    
                if now - metadata['timestamp'] > self.max_age:
                    stats['expired_items'] += 1
            except (pickle.PickleError, KeyError):
                stats['corrupted_items'] += 1
                
        return stats