"""Processor registry with metadata, categories, and search functionality.

Provides a flexible, extensible architecture for registering image processing operations
with rich metadata including categories, descriptions, and parameter schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class ProcessorParam:
    """Metadata for a processor parameter."""
    name: str
    type: str  # "float", "int", "bool", "str", "choice"
    default: Any
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None


@dataclass
class ProcessorInfo:
    """Metadata for a registered processor."""
    name: str
    func: Callable
    category: str
    description: str
    params: List[ProcessorParam] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class ProcessorRegistry:
    """Registry for image processing operations with categorization and search."""
    
    def __init__(self):
        self._processors: Dict[str, ProcessorInfo] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        category: str,
        description: str,
        params: Optional[List[ProcessorParam]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Decorator to register a processor function.
        
        Example:
            @processor_registry.register(
                name="gaussian_blur",
                category="Filters",
                description="Apply Gaussian blur to smooth images",
                params=[
                    ProcessorParam("sigma", "float", 1.0, "Blur strength", min=0.1, max=10.0)
                ],
                tags=["blur", "smooth", "denoise"]
            )
            def gaussian_blur(tile, coords, sigma=1.0):
                ...
        """
        def decorator(func: Callable) -> Callable:
            info = ProcessorInfo(
                name=name,
                func=func,
                category=category,
                description=description,
                params=params or [],
                tags=tags or [],
            )
            self._processors[name] = info
            
            # Update category index
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)
            
            return func
        return decorator
    
    def get(self, name: str) -> Optional[ProcessorInfo]:
        """Get processor info by name."""
        return self._processors.get(name)
    
    def get_func(self, name: str) -> Optional[Callable]:
        """Get processor function by name."""
        info = self._processors.get(name)
        return info.func if info else None
    
    def list_all(self) -> List[ProcessorInfo]:
        """List all registered processors."""
        return list(self._processors.values())
    
    def list_by_category(self, category: str) -> List[ProcessorInfo]:
        """List processors in a specific category."""
        names = self._categories.get(category, [])
        return [self._processors[name] for name in names if name in self._processors]
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(self._categories.keys())
    
    def search(self, query: str) -> List[ProcessorInfo]:
        """Search processors by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for info in self._processors.values():
            if (query_lower in info.name.lower() or
                query_lower in info.description.lower() or
                any(query_lower in tag.lower() for tag in info.tags)):
                results.append(info)
        return results
    
    def get_params(self, name: str) -> List[ProcessorParam]:
        """Get parameter schema for a processor."""
        info = self._processors.get(name)
        return info.params if info else []


# Global registry instance
processor_registry = ProcessorRegistry()
