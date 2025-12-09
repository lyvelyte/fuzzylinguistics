import pandas as pd
from fuzzylinguistics import setup_fls_from_data, DatasetConfig

# Define the paths to your data and config files
data_csv_path = "data.csv"
mf_json_path = "configs/membership_functions.json"

# Load the data from a single CSV file
df = pd.read_csv(data_csv_path)

input_df = df.iloc[:, :2]
output_df = df.iloc[:, 2:]

# Create a DatasetConfig object
car_dataset = DatasetConfig(
    category_name="Cars",
    model_name="Car Example",
    uses_qualifier=True,
    input_dimension_labels=list(input_df.columns),
    input_dimension_units=[None, None],
    input_data=input_df,      
    output_dimension_labels=list(output_df.columns),
    output_dimension_units=[None, None],
    output_data=output_df 
)

# Generate and print Fuzzy Linguistic Summaries.
fls = setup_fls_from_data([car_dataset], mf_json_path)
fls.generate_fls_one_model(results_dir = "output")
fls.print_results()