
from fuzzylinguistics import setup_fls

fls = setup_fls(config_json_path = "configs/config_simple_single.json")
fls.generate_fls_one_model(results_dir = "output")
fls.print_results()