# FuzzyLinguistics

[![PyPI version](https://badge.fury.io/py/fuzzylinguistics.svg)](https://badge.fury.io/py/fuzzylinguistics)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A powerful and flexible Python library for generating and simplifying fuzzy linguistic summaries of data. Turn complex datasets into human-readable, natural language insights.

This library implements a complete system for creating summaries of one or two datasets (e.g., comparing models), evaluating their truth and quality, and simplifying them from thousands of potential statements into a handful of concise, meaningful insights.

### Key Features

*   **Generate Summaries**: Create summaries for single datasets or compare two datasets against each other.
*   **Flexible Configuration**: Define your linguistic variables programmatically in Python or through a clean JSON interface.
*   **Powerful Simplification**: A multi-stage graph-based algorithm prunes and combines thousands of raw statements into a coherent, high-level summary.
*   **Quality Metrics**: Evaluates summaries based on truth, focus, simplicity, and relevance.
*   **Visualization**: Includes tools to plot membership functions and export the summary graph for analysis.

### Installation

Install `fuzzylinguistics` directly from PyPI:

```bash
pip install fuzzylinguistics
```

### Quick Start

The easiest way to get started is with a JSON configuration file.

**1. Create your configuration file (`config.json`)**

This file defines your data, linguistic variables (summarizers, qualifiers), and quantifiers.

```json
{
    "datasets": [
        {
            "category_name": "Cars",
            "model_name": "Car Example",
            "uses_qualifier": true,
            "input_dimension_labels": ["color"],
            "input_dimension_units": [null],
            "input_data": [
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1]
            ],
            "output_dimension_labels": ["speed"],
            "output_dimension_units": [null],
            "output_data": [
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1]
            ]
        }
    ],
    "summarizers": [
        {
            "dimension_name": "color",
            "predicates": ["Red", "Green", "Blue"],
            "trapezoidal_x_vals": [
                [0, 0, 0, 0],
                [0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1]
            ],
            "relevancy_weights": [1, 1, 1]
        }
    ],
    "qualifiers": [
        {
            "dimension_name": "speed",
            "predicates": ["Slow", "Normal", "Fast"],
            "trapezoidal_x_vals": [
                [0, 0, 0, 0],
                [0.5, 0.5, 0.5, 0.5],
                [1, 1, 1, 1]
            ],
            "relevancy_weights": [1, 1, 1]
        }
    ],
    "quantifiers": {
        "dimension_name": "quantifier",
        "predicates": ["None", "None", "A Few", "Some", "Many", "All"],
        "trapezoidal_x_vals": [
            [null, null, null, null],
            [0, 0, 0, 0],
            [0.00001, 0.00001, 0.00001, 0.5],
            [0.0, 0.5, 0.5, 0.99999],
            [0.5, 0.99999, 0.99999, 0.99999],
            [1, 1, 1, 1]
        ],
        "relevancy_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }
}
```

**2. Run the summarization in Python**

```python
# main.py
from fuzzylinguistics import setup_fls

# 1. Setup the system from the config file
fls = setup_fls(config_json_path="config.json")

# 2. Generate and simplify the summary
# Results will be saved to the specified directory
fls.generate_fls_one_model(results_dir='./car_summary_results')

# 3. Print the final, simplified summary
print("--- Final Simplified Summary ---")
for statement in fls.results["simplified_stage_4_summary"]:
    print(statement)
```

**Example Output:**

```
--- Final Simplified Summary ---
None are green color or normal speed.
Of the cars with blue color none are slow speed, and all are fast speed.
Of the cars with red color none are fast speed, and all are slow speed.
```

### Examples

For more detailed examples, including how to compare two models and how to set up the system programmatically without JSON, please see the `/examples` directory in the project repository.

### License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.