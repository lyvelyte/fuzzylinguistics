"""
fuzzylinguistics: A Python library for generating fuzzy linguistic summaries.
"""

from .fls_pydantic_interface import setup_fls, setup_fls_from_data, setup_fls_with_auto_partitions, DatasetConfig
from .fuzzy_linguistic_summaries import FuzzyLinguisticSummaries

# Define the package version dynamically from git tags
try:
    from ._version import version as __version__
except ImportError:
    # Fallback if package is not installed (e.g. local dev without setup)
    __version__ = "0.0.0+unknown"

# Define the public API for wildcard imports (from fuzzylinguistics import *)
__all__ = [
    'setup_fls',
    'setup_fls_from_data',
    'setup_fls_with_auto_partitions',
    'DatasetConfig',
    'FuzzyLinguisticSummaries'
]