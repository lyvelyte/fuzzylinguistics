from .fuzzy_linguistic_summaries import FuzzyLinguisticSummaries
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
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
    Now accepts pandas DataFrame or NumPy array for data fields.
    """
    category_name: str
    model_name: str
    uses_qualifier: bool
    input_dimension_labels: List[str]
    input_dimension_units: List[Optional[str]]
    input_data: Union[NpNDArray, pd.DataFrame, List]
    output_dimension_labels: List[str]
    output_dimension_units: List[Optional[str]]
    output_data: Union[NpNDArray, pd.DataFrame, List]

    class Config:
        arbitrary_types_allowed = True

    @validator('input_data', 'output_data', pre=True, each_item=False)
    def convert_to_numpy(cls, v):
        if isinstance(v, pd.DataFrame):
            return v.values
        if isinstance(v, list):
            return np.array(v)
        return v

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
    Sets up the FuzzyLinguisticSummaries instance from a full Pydantic configuration object in a JSON file.
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

def setup_fls_from_data(
    datasets: List[DatasetConfig],
    membership_functions_json_path: str
):
    """
    Sets up the FuzzyLinguisticSummaries instance from a list of DatasetConfig objects
    and a JSON file for membership functions.
    """
    assert os.path.exists(membership_functions_json_path), f"Error: Could not find file: {membership_functions_json_path}"
    with open(membership_functions_json_path, "r") as f:
        mf_config_dict = json.load(f)

    # Create the full configuration dictionary
    config_dict = {
        "datasets": [ds.dict() for ds in datasets],
        **mf_config_dict
    }

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

def _autogenerate_mfs(
    data: np.ndarray,
    dimension_labels: List[str],
    auto_configs: Optional[Dict[str, Dict]] = None
) -> List[MembershipFunctionConfig]:
    """
    Helper to generate membership function configs from data with granular control.
    """
    mf_configs = []
    if auto_configs is None:
        auto_configs = {}

    for i, dim_name in enumerate(dimension_labels):
        dimension_data = data[:, i]
        d_min, d_max = np.min(dimension_data), np.max(dimension_data)

        # Get the specific configuration for this dimension or use a default
        default_config = {"num_partitions": 4, "spacing": "linear"}
        config = auto_configs.get(dim_name, default_config)

        # Handle case where all data points are the same
        if np.isclose(d_min, d_max):
            print(f"Warning: All values in dimension '{dim_name}' are the same. Creating a single fuzzy set.")
            centers = np.array([d_min])
        # Case 1: Manual partitions are specified
        elif "manual_partitions" in config and config["manual_partitions"]:
            centers = np.array(sorted(config["manual_partitions"]))
            # Ensure the provided partitions are within the data bounds for safety
            if centers[0] < d_min or centers[-1] > d_max:
                 print(f"Warning: Manual partitions for '{dim_name}' are outside the data range [{d_min:.2f}, {d_max:.2f}].")
        # Case 2: Auto-generate partitions
        else:
            n = config.get("num_partitions", 4)
            spacing = config.get("spacing", "linear")
            
            # Add a small epsilon to avoid log(0) if d_min is 0 or negative
            d_min_safe = d_min if d_min > 0 else 1e-9

            if spacing == 'linear':
                centers = np.linspace(d_min, d_max, n)
            elif spacing == 'log-min':
                # Concentrates points near the minimum
                log_min = np.log(d_min_safe)
                log_max = np.log(d_max)
                centers = np.exp(np.linspace(log_min, log_max, n))
            elif spacing == 'log-max':
                # Concentrates points near the maximum
                log_min = np.log(d_min_safe)
                log_max = np.log(d_max)
                # Generate as if concentrated near min, then flip the distribution
                temp_centers = np.exp(np.linspace(log_min, log_max, n))
                centers = d_max - temp_centers[::-1] + d_min
            else:
                raise ValueError(f"Invalid spacing '{spacing}' for dimension '{dim_name}'. Must be 'linear', 'log-min', or 'log-max'.")

        # Generate trapezoidal values for triangular membership functions
        trapezoids = []
        for j, center in enumerate(centers):
            # For boundaries, use the actual data min/max
            left = centers[j-1] if j > 0 else d_min
            right = centers[j+1] if j < len(centers) - 1 else d_max
            trapezoids.append([left, center, center, right])

        predicates = [f"about {c:.2f}" for c in centers]
        relevancy_weights = [1.0] * len(centers)

        mf_configs.append(
            MembershipFunctionConfig(
                dimension_name=dim_name,
                predicates=predicates,
                trapezoidal_x_vals=np.array(trapezoids),
                relevancy_weights=relevancy_weights
            )
        )
    return mf_configs


def setup_fls_with_auto_partitions(
    datasets: List[DatasetConfig],
    summarizer_auto_configs: Optional[Dict[str, Dict]] = None,
    qualifier_auto_configs: Optional[Dict[str, Dict]] = None,
):
    """
    Sets up the FLS instance by automatically generating membership functions
    based on the provided data and flexible, per-dimension configurations.

    Args:
        datasets: A list of DatasetConfig objects containing the data.
        summarizer_auto_configs (dict, optional): Configuration for input dimensions (summarizers).
            Example:
            {
                "dimension_name_1": {"num_partitions": 5, "spacing": "linear"},
                "dimension_name_2": {"spacing": "log-min", "num_partitions": 4},
                "dimension_name_3": {"manual_partitions": [10, 20, 50, 100]}
            }
        qualifier_auto_configs (dict, optional): Configuration for output dimensions (qualifiers).
            Follows the same structure as summarizer_auto_configs.
    """
    fls = FuzzyLinguisticSummaries()

    # --- 1. Add data categories ---
    all_input_data, all_output_data = [], []
    qualifiers_needed = False
    for dataset in datasets:
        fls.add_data_category(
            dataset.category_name, dataset.uses_qualifier, dataset.input_data,
            dataset.input_dimension_labels, dataset.input_dimension_units,
            dataset.output_data, dataset.output_dimension_labels, dataset.output_dimension_units,
            dataset.model_name
        )
        all_input_data.append(dataset.input_data)
        if dataset.uses_qualifier:
            qualifiers_needed = True
            all_output_data.append(dataset.output_data)

    # --- 2. Generate and add summarizers ---
    combined_input_data = np.vstack(all_input_data)
    summarizer_configs = _autogenerate_mfs(
        combined_input_data,
        datasets[0].input_dimension_labels,
        summarizer_auto_configs
    )
    for summarizer in summarizer_configs:
        fls.add_summarizer(
            summarizer.dimension_name, summarizer.dimension_name, summarizer.predicates,
            summarizer.trapezoidal_x_vals, summarizer.relevancy_weights
        )
        print(f"Generated {len(summarizer.predicates)} summarizers for dimension '{summarizer.dimension_name}'.")

    # --- 3. Generate and add qualifiers (if needed) ---
    if qualifiers_needed and all_output_data:
        combined_output_data = np.vstack(all_output_data)
        qualifier_configs = _autogenerate_mfs(
            combined_output_data,
            datasets[0].output_dimension_labels,
            qualifier_auto_configs
        )
        for qualifier in qualifier_configs:
            fls.add_qualifier(
                qualifier.dimension_name, qualifier.dimension_name, qualifier.predicates,
                qualifier.trapezoidal_x_vals, qualifier.relevancy_weights
            )
            print(f"Generated {len(qualifier.predicates)} qualifiers for dimension '{qualifier.dimension_name}'.")

    # --- 4. Add static quantifiers ---
    quantifiers_config = { "predicates": ["None", "None", "A Few", "Some", "Many", "All"], "trapezoidal_x_vals": [[np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0], [0.00001, 0.00001, 0.2, 0.5], [0.2, 0.5, 0.5, 0.8], [0.5, 0.8, 1.0, 1.0], [1, 1, 1, 1]], "relevancy_weights": [1.0] * 6 }
    fls.add_quantifiers(quantifiers_config["predicates"], np.array(quantifiers_config["trapezoidal_x_vals"]), quantifiers_config["relevancy_weights"])
    print("Added static quantifiers.")

    return fls