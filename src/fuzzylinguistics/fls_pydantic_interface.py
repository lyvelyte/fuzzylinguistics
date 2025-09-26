from .fuzzy_linguistic_summaries import FuzzyLinguisticSummaries
import numpy as np
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from pydantic_numpy.typing import NpNDArray
import json
import os

class MembershipFunctionConfig(BaseModel):
    """
    Configuration for a single fuzzy membership function.
    """
    dimension_name: str
    predicates: List[str]
    trapezoidal_x_vals: NpNDArray
    relevancy_weights: List[float]

    @validator('trapezoidal_x_vals', pre=True)
    def replace_none_with_nan(cls, v):
        """
        Recursively replaces None values in a list of lists with float('nan').
        This allows using 'null' in the JSON configuration.
        """
        if isinstance(v, list):
            return [[float('nan') if item is None else item for item in sublist] for sublist in v]
        return v

class DatasetConfig(BaseModel):
    """
    Configuration for a single dataset.
    """
    category_name: str
    model_name: str
    uses_qualifier: bool
    input_dimension_labels: List[str]
    input_dimension_units: List[Optional[str]]
    input_data: NpNDArray
    output_dimension_labels: List[str]
    output_dimension_units: List[Optional[str]]
    output_data: NpNDArray

class FuzzySystemConfig(BaseModel):
    """
    Overall configuration for the fuzzy linguistic summary system.
    """
    summarizers: List[MembershipFunctionConfig]
    qualifiers: List[MembershipFunctionConfig]
    quantifiers: MembershipFunctionConfig
    datasets: List[DatasetConfig]

def setup_fls(config_json_path: str):
    """
    Sets up the FuzzyLinguisticSummaries instance from a Pydantic configuration object.
    """

    assert os.path.exists(config_json_path), f"Error: Could not find file: {config_json_path}"
    with open(config_json_path, "r") as f:
        config_dict = json.load(f)
    
    config = FuzzySystemConfig(**config_dict)

    fls = FuzzyLinguisticSummaries()

    # Add data categories
    for dataset in config.datasets:
        fls.add_data_category(
            dataset.category_name,
            dataset.uses_qualifier,
            dataset.input_data,
            dataset.input_dimension_labels,
            dataset.input_dimension_units,
            dataset.output_data,
            dataset.output_dimension_labels,
            dataset.output_dimension_units,
            dataset.model_name
        )

    # Add summarizers
    for summarizer in config.summarizers:
        fls.add_summarizer(
            summarizer.dimension_name,
            summarizer.dimension_name,
            summarizer.predicates,
            summarizer.trapezoidal_x_vals,
            summarizer.relevancy_weights
        )

    # Add qualifiers
    for qualifier in config.qualifiers:
        fls.add_qualifier(
            qualifier.dimension_name,
            qualifier.dimension_name,
            qualifier.predicates,
            qualifier.trapezoidal_x_vals,
            qualifier.relevancy_weights
        )

    # Add quantifiers
    fls.add_quantifiers(
        config.quantifiers.predicates,
        config.quantifiers.trapezoidal_x_vals,
        config.quantifiers.relevancy_weights
    )

    return fls
