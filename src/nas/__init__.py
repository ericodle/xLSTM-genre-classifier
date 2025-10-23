"""
Neural Architecture Search (NAS) module for GenreDiscern.

This module provides comprehensive NAS capabilities including:
- Differentiable Architecture Search (DARTS)
- Evolutionary Architecture Search
- Random Architecture Search
- Multi-objective optimization
- Architecture performance analysis
"""

from .search_space import SearchSpace, Architecture
from .search_algorithms import DARTSSearcher, EvolutionarySearcher, RandomSearcher
from .evaluator import ArchitectureEvaluator
from .nas_runner import NASRunner

__all__ = [
    'SearchSpace',
    'Architecture', 
    'DARTSSearcher',
    'EvolutionarySearcher',
    'RandomSearcher',
    'ArchitectureEvaluator',
    'NASRunner'
]
