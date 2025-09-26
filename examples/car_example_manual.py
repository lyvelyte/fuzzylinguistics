
from fuzzylinguistics import FuzzyLinguisticSummaries
import numpy as np

def generate_data_less_simple_single(fls):
    input_dimension_labels = ["color", "age"]
    input_dimension_units = [None, None]
    
    # 10 Samples - 0 = Red, 0.5 = Green, 1 = Blue
    #              0 = Old, 0.5 = Modern, 1 = Future
    input_data = np.array([
        [0, 0], # 1
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0], # 5
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1], # 10
    ])

    output_dimension_labels = ["speed", "size"]
    output_dimension_units = [None, None]
    
    # 10 Samples -  0 = Slow, 0.5 = Normal, 1 = Fast
    #               0 = Small, 0.5 = Medium, 1 = Big
    output_data = np.array([
        [0, 0], # 1
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0], # 5
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1], # 10
    ])

    model_name = "Car Example"

    uses_qualifier = True

    category_name = "Cars"

    fls.add_data_category(category_name, uses_qualifier, input_data, input_dimension_labels, input_dimension_units, output_data, output_dimension_labels, output_dimension_units, model_name)

    # =================== Setup summarizer (P) attributes + fuzzy predicate membership functions. =====================
    fuzzy_predicates = ['Red', 'Green', 'Blue']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_summarizer(input_dimension_labels[0], input_dimension_labels[0], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    fuzzy_predicates = ['Old', 'Modern', 'Future']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_summarizer(input_dimension_labels[1], input_dimension_labels[1], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    # =================== Setup qualifier (R) attributes + fuzzy predicate membership functions. =====================
    fuzzy_predicates = ['Slow', 'Normal', 'Fast']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_qualifier(output_dimension_labels[0], output_dimension_labels[0], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    fuzzy_predicates = ['Small', 'Medium', 'Big']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_qualifier(output_dimension_labels[1], output_dimension_labels[1], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    # =================== Setup quantifier (Q) attribute + fuzzy predicate membership function. =====================
    fuzzy_predicates = ['None', 'None', 'A Few', 'Some', 'Many', "All"]
    espilon = 10e-6
    NaN = float('nan')
    trapezoidal_fuzzy_membership_function_x_vals = [[NaN, NaN, NaN, NaN], 
                                                    [0, 0, 0, 0],
                                                    [espilon, espilon, espilon, 0.5], 
                                                    [0.0, 0.5, 0.5, 1-espilon], 
                                                    [0.5, 1-espilon, 1-espilon, 1-espilon],
                                                    [1, 1, 1, 1]]
    operational_relevancy_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    fls.add_quantifiers(fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)


def generate_data_simple_single(fls):
    input_dimension_labels = ["color"]
    input_dimension_units = [None]
    
    # 20 Samples - 0 = Red Cars, 0.5 = Green, 1 = Blue Cars
    input_data = np.array([
        [0], # 1
        [0],
        [0],
        [0],
        [0], # 5
        [0],
        [0],
        [0],
        [0],
        [1], # 10
    ])

    output_dimension_labels = ["speed"]
    output_dimension_units = [None]
    
    # 20 Samples - 0 = Slow, 0.5 = Normal, 1 = Fast
    output_data = np.array([
        [0], # 1
        [0],
        [0],
        [0],
        [0], # 5
        [0],
        [0],
        [0],
        [0],
        [1], # 10
    ])

    model_name = "Car Example"

    uses_qualifier = True

    category_name = "Cars"

    fls.add_data_category(category_name, uses_qualifier, input_data, input_dimension_labels, input_dimension_units, output_data, output_dimension_labels, output_dimension_units, model_name)

    # =================== Setup summarizer (P) attributes + fuzzy predicate membership functions. =====================
    fuzzy_predicates = ['Red', 'Green', 'Blue']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_summarizer(input_dimension_labels[0], input_dimension_labels[0], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    # =================== Setup qualifier (R) attributes + fuzzy predicate membership functions. =====================
    fuzzy_predicates = ['Slow', 'Normal', 'Fast']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_qualifier(output_dimension_labels[0], output_dimension_labels[0], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    # =================== Setup quantifier (Q) attribute + fuzzy predicate membership function. =====================
    fuzzy_predicates = ['None', 'None', 'A Few', 'Some', 'Many', "All"]
    espilon = 10e-6
    NaN = float('nan')
    trapezoidal_fuzzy_membership_function_x_vals = [[NaN, NaN, NaN, NaN], 
                                                    [0, 0, 0, 0],
                                                    [espilon, espilon, espilon, 0.5], 
                                                    [0.0, 0.5, 0.5, 1-espilon], 
                                                    [0.5, 1-espilon, 1-espilon, 1-espilon],
                                                    [1, 1, 1, 1]]
    operational_relevancy_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    fls.add_quantifiers(fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

def generate_data_simple_two_modles(fls):
    input_dimension_labels = ["color"]
    input_dimension_units = [None]
    
    # 20 Samples - 0 = Red Cars, 0.5 = Green, 1 = Blue Cars
    input_data_A = np.array([
        [0], # 1
        [0],
        [0],
        [0],
        [0], # 5
        [0],
        [0],
        [0],
        [0],
        [1], # 10
    ])

    input_data_B = np.array([
        [0.5], # 1
        [0.5],
        [0.5],
        [1],
        [0], # 5
        [0],
        [0],
        [0],
        [0],
        [0], # 10
    ])

    output_dimension_labels = ["speed"]
    output_dimension_units = [None]
    
    # 20 Samples - 0 = Slow, 0.5 = Normal, 1 = Fast
    output_data_A = np.array([
        [0], # 1
        [0],
        [0],
        [0],
        [0], # 5
        [0],
        [0],
        [0],
        [0],
        [1], # 10
    ])

    output_data_B = np.array([
        [1], # 1
        [1],
        [1],
        [1],
        [0], # 5
        [0],
        [0],
        [0],
        [0],
        [0], # 10
    ])

    model_name_A = "Ford"
    model_name_B = "Chevy"

    uses_qualifier = True

    category_name = "cars"

    fls.add_data_category(category_name, uses_qualifier, input_data_A, input_dimension_labels, input_dimension_units, output_data_A, output_dimension_labels, output_dimension_units, model_name_A)
    fls.add_data_category(category_name, uses_qualifier, input_data_B, input_dimension_labels, input_dimension_units, output_data_B, output_dimension_labels, output_dimension_units, model_name_B)

    # =================== Setup summarizer (P) attributes + fuzzy predicate membership functions. =====================
    fuzzy_predicates = ['Red', 'Green', 'Blue']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_summarizer(input_dimension_labels[0], input_dimension_labels[0], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    # =================== Setup qualifier (R) attributes + fuzzy predicate membership functions. =====================
    fuzzy_predicates = ['Slow', 'Normal', 'Fast']
    trapezoidal_fuzzy_membership_function_x_vals = [[0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]
    operational_relevancy_weights = [1, 1, 1]
    fls.add_qualifier(output_dimension_labels[0], output_dimension_labels[0], fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

    # =================== Setup quantifier (Q) attribute + fuzzy predicate membership function. =====================
    fuzzy_predicates = ['None', 'Many less', 'Some less', 'Few less', "The Same Amount", 'Few more', 'Some more', 'Many more']
    espilon = 10e-6
    NaN = float('nan')
    trapezoidal_fuzzy_membership_function_x_vals = [[NaN, NaN, NaN, NaN],
                                                    [-1.0, -1.0, -1.0, -0.5],
                                                    [-1.0, -0.5, -0.5, -espilon],
                                                    [-0.5, -espilon, -espilon, -espilon],
                                                    [0.0, 0.0, 0.0, 0.0],
                                                    [espilon, espilon, espilon, 0.5], 
                                                    [espilon, 0.5, 0.5, 1.0], 
                                                    [0.5, 1.0, 1.0, 1.0]]
    
    operational_relevancy_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    fls.add_quantifiers(fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)

if __name__ == '__main__':
    fls = FuzzyLinguisticSummaries()

    compare_two_models = False
    if compare_two_models:
        generate_data_simple_two_modles(fls)
        results_dir = r'D:\FLS_results\car_example\two_models'
        results = fls.generate_fls_two_models(results_dir)
    else:
        generate_data_simple_single(fls)
        # generate_data_less_simple_single(fls)
        results_dir = r'D:\FLS_results\car_example\single_model'
        results = fls.generate_fls_one_model(results_dir)

    # Print initial summary:
    print("\nPrinting initial summary:")
    for i in range(len(results["initial_linguistic_summary"]["linguistic_statements"])):
        s = results["initial_linguistic_summary"]["linguistic_statements"][i]
        t = results["initial_linguistic_summary"]["truth_vals"][i]
        # if t > 0:
        print(f"{t:0.3} | {s}")

    # Print intermediate simplified summary:
    print("\nPrinting intermediate simplified summary:")
    for s in results["simplified_stage_3_summary"]:
        print(s)

    # Print final simplified summary:
    print("\nPrinting final simplified summary:")
    for s in results["simplified_stage_4_summary"]:
        print(s)