"""
fuzzylinguistics: A Python library for generating fuzzy linguistic summaries.
"""

from .fls_pydantic_interface import setup_fls
from .fuzzy_linguistic_summaries import FuzzyLinguisticSummaries

# Define the package version
__version__ = "0.1.0"

# Define the public API for wildcard imports (from fuzzylinguistics import *)
__all__ = [
    'setup_fls',
    'FuzzyLinguisticSummaries'
]