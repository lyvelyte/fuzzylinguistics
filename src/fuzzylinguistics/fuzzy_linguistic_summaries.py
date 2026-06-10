import numpy as np
# import cupy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import csv
import networkx as nx
import colorsys
from itertools import product
import json
import time
import statistics
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import re
from collections import Counter
import math

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

import re
from typing import List, Optional, Sequence, Tuple

def combine_statements(
    statements: List[str],
    fuzzy_predicates: List[str]
) -> List[str]:
    """
    Combine statements sharing the same subject (prefix) and quantifier,
    using strict prefix matching (case-insensitive, whitespace-normalized).
    Keeps unmatched statements intact.
    """
    # CRITICAL FIX: Ensure the list of predicates is unique by converting to a set.
    # Then sort by length (longest first) to ensure "many less" is checked before "less".
    unique_predicates = sorted(
        list(set(q.strip().lower() for q in fuzzy_predicates if q.strip())),
        key=lambda x: -len(x)
    )

    groups = {}

    for s in statements:
        s_clean = s.strip().rstrip(".")
        s_lower = s_clean.lower()

        matched = False
        # Iterate through the unique, sorted predicates
        for q in unique_predicates:
            pattern = r"\b" + re.escape(q) + r"\b"
            m = re.search(pattern, s_lower)

            if m and " are " in s_lower[m.start():]:
                start = m.start()
                prefix = s_clean[:start].strip()
                rest = s_clean[start:]

                try:
                    _, cond = rest.split(" are ", 1)
                except ValueError:
                    continue

                condition = cond.strip()
                key = re.sub(r"\s+", " ", prefix.lower())

                if key not in groups:
                    groups[key] = {"subject": prefix, "clauses": {}}

                # Use the short, unique predicate 'q' as the key
                groups[key]["clauses"].setdefault(q, []).append(condition)

                matched = True
                break

        if not matched:
            key = re.sub(r"\s+", " ", s_clean.lower())
            if key not in groups:
                groups[key] = {"subject": s_clean, "clauses": {}}

    # Rebuild final combined statements
    final = []
    for g in sorted(groups.values(), key=lambda x: x["subject"].lower()):
        prefix = g["subject"]
        clauses = []

        # Iterate through the unique predicates again to build the final sentence
        for q in unique_predicates:
            if q in g["clauses"]:
                seen = set()
                uniq_conds = [
                    c for c in g["clauses"][q]
                    if not (c.lower() in seen or seen.add(c.lower()))
                ]

                joiner = " or " if "none" in q else " and "
                clauses.append(f"{q} are {joiner.join(uniq_conds)}")

        if clauses:
            combined_clauses = ', and '.join(clauses)
            sentence = f"{prefix} {combined_clauses}" if prefix else combined_clauses
        else:
            sentence = g["subject"]

        sentence = sentence.strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith("."):
                sentence += "."
            final.append(sentence)

    return final



def find_partitions(list_a, list_b):
    """
    Partitions two lists of equal length into two groups based on a specific rule.

    Args:
        list_a: A list of integers describing one attribute type.
        list_b: A list of integers describing another attribute type.

    Returns:
        A tuple containing two lists of indices, representing the two partitions.
        Returns ([], []) if the lists are empty.
    """
    if not list_a:
        return [], []

    # Handle the case where all elements belong to a single group first
    if all(val == list_a[0] for val in list_a):
        # All of list A has the same value, so this is Group 1
        return list(range(len(list_a))), []

    if all(val == list_b[0] for val in list_b):
        # All of list B has the same value, so this is Group 2
        return [], list(range(len(list_a)))

    # Find the most common value in each list
    a_counts = Counter(list_a)
    b_counts = Counter(list_b)

    # The value that defines a group will appear more than once
    a_mode = a_counts.most_common(1)[0][0]
    b_mode = b_counts.most_common(1)[0][0]

    group1_indices = []
    group2_indices = []

    # Identify the value in list A that corresponds to Group 1
    # Group 1 has the same 'a' value and different 'b' values
    group1_defining_value = a_mode

    # Identify the value in list B that corresponds to Group 2
    # Group 2 has the same 'b' value and different 'a' values
    group2_defining_value = b_mode

    for i in range(len(list_a)):
        if list_a[i] == group1_defining_value:
            group1_indices.append(i)
        elif list_b[i] == group2_defining_value:
            group2_indices.append(i)

    return group1_indices, group2_indices

def scale_axis(ax, all_xs, all_ys, tick_size):
    # Use nanmin and nanmax to safely handle NaN values
    min_x = np.nanmin(all_xs)
    max_x = np.nanmax(all_xs)
    min_y = np.nanmin(all_ys)
    max_y = np.nanmax(all_ys)

    # Check if all values were NaN, in which case min/max will be NaN
    if math.isnan(min_x) or math.isnan(max_x) or math.isnan(min_y) or math.isnan(max_y):
        # Handle the case where there is no data to scale, perhaps by setting default limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        pad_x = 0.10 * abs(max_x - min_x)
        pad_y = 0.10 * abs(max_y - min_y)
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)

def print_stats(stage_num, data):
    if len(data) > 0:
        average = sum(data) / len(data)  # or use statistics.mean(float_list)
        minimum = min(data)
        maximum = max(data)
    else:
        average = 0
        minimum = 0
        maximum = 0

    if len(data) > 1:
        std_dev = statistics.stdev(data)  # Use statistics.pstdev for population standard deviation
    else:
        std_dev = 0

    print(f"Stage_{stage_num}: avg = {average}, min = {minimum}, max = {maximum}, std_dev = {std_dev}")

def scale_array_to_01(arr):
    """
    Scales a NumPy array to the range [0, 1].

    Args:
        arr: The input NumPy array.

    Returns:
        The scaled NumPy array.
    """
    arr_min = arr.min()
    arr_max = arr.max()
    arr_range = arr_max - arr_min
    if arr_range == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / arr_range

def add_angle_offset(base_angle, offset):
    """
    Adds an offset to a base angle and normalizes the result to the range [0, 360).

    Args:
        base_angle: The starting angle in degrees.
        offset: The angle offset to add, in degrees.

    Returns:
        The new angle, normalized to be within the range [0, 360).
    """
    return (base_angle + offset) % 360

 # Find the root node (node with no incoming edges)
def find_root_node(nx_graph):
    for node in nx_graph.nodes:
        if nx_graph.in_degree(node) == 0:  # No incoming edges
            return node
    raise ValueError("No root node found! Is the graph a DAG?")

def apply_hierarchical_layout(nx_graph, root_node, spacing=100):
    """Assign positions to nodes for a hierarchical layout."""
    levels = nx.single_source_shortest_path_length(nx_graph, root_node)
    pos = {}
    level_nodes = {}

    # Group nodes by their level
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    # Assign positions level by level
    for level, nodes in level_nodes.items():
        y_pos = -level * spacing * 3
        x_step = spacing * 2
        start_x = -(len(nodes) - 1) * x_step / 2
        for i, node in enumerate(nodes):
            pos[node] = {'x': start_x + i * x_step, 'y': y_pos, 'z': 0.0}

    # Update node positions in the graph
    for node_id, position in pos.items():
        nx_graph.nodes[node_id]['viz']['position'] = position

    return nx_graph

def convert_numpy_strings_to_str(nx_graph):
    # Convert node attributes
    for node, data in nx_graph.nodes(data=True):
        for attr_key, attr_val in data.items():
            if isinstance(attr_val, np.str_):
                data[attr_key] = str(attr_val)

    # Convert edge attributes
    for u, v, data in nx_graph.edges(data=True):
        for attr_key, attr_val in data.items():
            if isinstance(attr_val, np.str_):
                data[attr_key] = str(attr_val)

    # Convert graph-level attributes if any
    for attr_key, attr_val in nx_graph.graph.items():
        if isinstance(attr_val, np.str_):
            nx_graph.graph[attr_key] = str(attr_val)

    return nx_graph

def convert_dicts_to_str(nx_graph):
    # Convert node attributes
    for node, data in nx_graph.nodes(data=True):
        for attr_key, attr_val in list(data.items()):  # list() to avoid RuntimeError during iteration
            if isinstance(attr_val, dict):
                data[attr_key] = json.dumps(attr_val)

    # Convert edge attributes
    for u, v, data in nx_graph.edges(data=True):
        for attr_key, attr_val in list(data.items()):
            if isinstance(attr_val, dict):
                data[attr_key] = json.dumps(attr_val)

    # Convert graph-level attributes if any
    for attr_key, attr_val in list(nx_graph.graph.items()):
        if isinstance(attr_val, dict):
            nx_graph.graph[attr_key] = json.dumps(attr_val)

    return nx_graph

_CUPY_PROBE_RESULT: Optional[Tuple[bool, str]] = None


def _probe_cupy_available() -> Tuple[bool, str]:
    """Return whether CuPy can initialize a CUDA device without crashing."""
    global _CUPY_PROBE_RESULT
    if _CUPY_PROBE_RESULT is not None:
        return _CUPY_PROBE_RESULT

    probe = (
        "import cupy as cp\n"
        "count = cp.cuda.runtime.getDeviceCount()\n"
        "cp.cuda.runtime.free(0)\n"
        "print(count)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        _CUPY_PROBE_RESULT = False, str(exc)
        return _CUPY_PROBE_RESULT

    if result.returncode != 0:
        message = (result.stderr or result.stdout or "CuPy probe failed").strip()
        _CUPY_PROBE_RESULT = False, message
        return _CUPY_PROBE_RESULT

    try:
        device_count = int((result.stdout or "0").strip().splitlines()[-1])
    except (IndexError, ValueError):
        device_count = 0

    if device_count <= 0:
        _CUPY_PROBE_RESULT = False, "No CUDA devices reported by CuPy"
    else:
        _CUPY_PROBE_RESULT = True, f"{device_count} CUDA device(s) available"
    return _CUPY_PROBE_RESULT


def _resolve_array_backend(compute_backend: str):
    backend = compute_backend.lower()
    if backend not in {"auto", "numpy", "cupy"}:
        raise ValueError(
            "compute_backend must be one of 'auto', 'numpy', or 'cupy'."
        )

    if backend == "numpy":
        return np, "numpy"

    cupy_available, message = _probe_cupy_available()
    if not cupy_available:
        if backend == "cupy":
            raise RuntimeError(
                "compute_backend='cupy' was requested, but CuPy could not "
                f"initialize a CUDA device: {message}"
            )
        print(f"CuPy backend unavailable ({message}); falling back to NumPy.")
        return np, "numpy"

    try:
        import cupy as cp
    except Exception as exc:
        if backend == "cupy":
            raise RuntimeError(
                "compute_backend='cupy' was requested, but CuPy could not be "
                f"imported: {exc}"
            ) from exc
        print(f"CuPy import failed ({exc}); falling back to NumPy.")
        return np, "numpy"

    return cp, "cupy"


def _to_numpy(array):
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def evaluate_trapezoidal_fuzzy_membership_backend(trapezoid_points, x, xp):
    x = xp.asarray(x)
    memberships = xp.zeros_like(x, dtype=float)
    trapezoid_points = xp.asarray(trapezoid_points, dtype=float)

    # --- special case: all NaN ---
    if bool(xp.all(xp.isnan(x))) and bool(xp.all(xp.isnan(trapezoid_points))):
        memberships = xp.ones_like(x, dtype=float)
        return memberships

    # --- normal trapezoid case ---
    a, b, c, d = [trapezoid_points[i] for i in range(4)]

    if bool(b != a):
        indx = xp.logical_and(x >= a, x < b)
        memberships[indx] = (x[indx] - a) / (b - a)

    indx = xp.logical_and(x >= b, x <= c)
    memberships[indx] = 1.0

    if bool(d != c):
        indx = xp.logical_and(x > c, x <= d)
        memberships[indx] = (d - x[indx]) / (d - c)

    return memberships


def evaluate_trapezoidal_fuzzy_membership(trapezoid_points, x):
    return evaluate_trapezoidal_fuzzy_membership_backend(trapezoid_points, x, np)


class StatementIndexSpace:
    """Lazy mixed-radix view of deterministic statement index products."""

    def __init__(
        self,
        predicate_counts: Sequence[int],
        materialize_limit_bytes: int = 64 * 1024 * 1024,
    ):
        self.predicate_counts = [int(count) for count in predicate_counts]
        self.bases = np.array([count + 1 for count in self.predicate_counts], dtype=int)
        self.n_dimensions = len(self.predicate_counts)
        self.total = int(np.prod(self.bases)) if self.n_dimensions else 1
        self.shape = (self.total, self.n_dimensions)

        divisors = []
        divisor = 1
        for base in reversed(self.bases.tolist()):
            divisors.append(divisor)
            divisor *= base
        self.divisors = np.array(list(reversed(divisors)), dtype=np.int64)

        materialized_bytes = self.total * max(self.n_dimensions, 1) * 8
        if materialized_bytes <= materialize_limit_bytes:
            self._array = self.decode_batch(0, self.total)
        else:
            self._array = None

    def decode_one(self, index: int) -> np.ndarray:
        return self.decode_batch(index, index + 1)[0]

    def decode_batch(self, start: int, stop: int) -> np.ndarray:
        if self.n_dimensions == 0:
            return np.empty((stop - start, 0), dtype=int)
        rows = np.arange(start, stop, dtype=np.int64)[:, None]
        digits = (rows // self.divisors[None, :]) % self.bases[None, :]
        return digits.astype(int) - 1

    def encode(self, values: Sequence[int]) -> int:
        if len(values) != self.n_dimensions:
            raise ValueError("Statement code has the wrong number of dimensions.")
        if self.n_dimensions == 0:
            return 0
        digits = np.asarray(values, dtype=np.int64) + 1
        if np.any(digits < 0) or np.any(digits >= self.bases):
            raise ValueError("Statement code contains an invalid predicate index.")
        return int(np.sum(digits * self.divisors))

    def __getitem__(self, key):
        if self._array is not None:
            return self._array[key]
        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, (int, np.integer)):
                return self.decode_one(int(row_key))[col_key]
            rows = np.asarray(row_key)
            decoded = np.vstack([self.decode_one(int(row)) for row in rows])
            return decoded[:, col_key]
        if isinstance(key, (int, np.integer)):
            return self.decode_one(int(key))
        rows = np.asarray(key)
        return np.vstack([self.decode_one(int(row)) for row in rows])

class FuzzyLinguisticData:
    def __init__(self, index, category_name, uses_qualifier, input_data, input_dimension_labels, input_dimension_units, output_data, output_dimension_labels, output_dimension_units, model_name):
        self.index = index

        self.model_name = model_name

        self.category_name = category_name

        self.input_data = input_data
        self.input_dimension_labels = input_dimension_labels
        self.input_dimension_units = input_dimension_units

        self.output_data = output_data
        self.output_dimension_labels = output_dimension_labels
        self.output_dimension_units = output_dimension_units

        if input_data.shape[0] != output_data.shape[0]:
            raise(ValueError(f"Error: Input and output data must have the same number of records. (input_data.shape[0] = {input_data.shape[0]}, output_data.shape[0] = {output_data.shape[0]})"))

        self.n_records = input_data.shape[0]

        self.uses_qualifier = uses_qualifier

        self.mu_p_intermediate = None
        self.mu_p = None
        self.L_p = None
        self.w_p = None
        self.B_p = None

        self.mu_r_intermediate = None
        self.mu_r = None
        self.L_r = None
        self.w_r = None
        self.B_r = None

class FuzzyLinguisticComponent:
    def __init__(self, attribute_name, display_name, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights):
        self.attribute_name = attribute_name
        self.display_name = display_name
        self.fuzzy_predicates = fuzzy_predicates
        self.trapezoidal_fuzzy_membership_function_x_vals = trapezoidal_fuzzy_membership_function_x_vals
        self.operational_relevancy_weights = operational_relevancy_weights

class FuzzyLinguisticSummaries:
    def __init__(self):
        self.params = {
            "LANGUAGE_MODE": 1,                         # 1 = Compare
            "USE_NEGATIVE_STATEMENTS": False,           # Enable to simplfy some compound statments by using "except from" when there's fewer siblings not being reported than being reported.
            "REMOVE_CHILDREN_FROM_TRUE_PARENTS": True,  # Enable to prune all descendents of parents who's truth value meets the user specified threshold for reporting.
        }

        self.data_categories = []

        self.all_statements_data_types = []
        self.all_statements_cnt = []
        self.all_statements_after_stage_1_cnt = []
        self.all_statements_after_stage_2_cnt = []
        self.all_statements_after_stage_3_cnt = []
        self.all_statements_after_stage_4_cnt = []

        self.summarizers = []
        self.max_summarizer_fuzzy_predicates = 0
        self.n_summarizers = 0

        self.qualifiers = []
        self.max_qualifier_fuzzy_predicates = 0
        self.n_qualifiers = 0

        self.quantifiers = None

        self.w_focus = 0.0
        self.w_complexity = 0.0

        self.model_names = []
        self.xp = np
        self.active_backend = "numpy"
        self.compute_backend = "auto"
        self.max_working_memory_mb = 1024
        self._max_working_memory_bytes = 1024 * 1024 * 1024
        self.statement_batch_size = None
        self.record_batch_size = None
        self.initial_summary_mode = "auto"
        self.max_initial_summary_items = 100_000
        self.initial_summary_streamed = False
        self._memmap_dir = None
        self._aggregated_scaled_indexes = set()

    def add_summarizer(self, attribute_name, display_name, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights):
        self.summarizers.append(FuzzyLinguisticComponent(attribute_name, display_name, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights))

        if len(fuzzy_predicates) > self.max_summarizer_fuzzy_predicates:
            self.max_summarizer_fuzzy_predicates = len(fuzzy_predicates)

        self.n_summarizers += 1

    def add_qualifier(self, attribute_name, display_name, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights):
        self.qualifiers.append(FuzzyLinguisticComponent(attribute_name, display_name, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights))

        if len(fuzzy_predicates) > self.max_qualifier_fuzzy_predicates:
            self.max_qualifier_fuzzy_predicates = len(fuzzy_predicates)

        self.n_qualifiers += 1

    def add_quantifiers(self, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights):
        if self.quantifiers is None:
            self.quantifiers = FuzzyLinguisticComponent(None, None, fuzzy_predicates, trapezoidal_fuzzy_membership_function_x_vals, operational_relevancy_weights)
        else:
            raise(ValueError("Error: Only one set of quantifiers is allowed and they have already been added."))

        self.n_quantifier_fuzzy_predicates = len(fuzzy_predicates)

    def add_data_category(self, category_name, uses_qualifier, input_data, input_dimension_labels, input_dimension_units, output_data, output_dimension_labels, output_dimension_units, model_name):

        if input_data.shape[0] > 0:
            if model_name not in self.model_names:
                self.model_names.append(model_name)
            self.data_categories.append(FuzzyLinguisticData(len(self.data_categories), category_name, uses_qualifier, input_data, input_dimension_labels, input_dimension_units, output_data, output_dimension_labels, output_dimension_units, model_name))
            print(f"Created data category. Model = {model_name} Name = {category_name}, input_points.shape = {input_data.shape}, output_data.shape  = {output_data.shape}")
        else:
            print(f"Did not create data category = {category_name} because input_data.shape[0] == 0.")

    def _configure_execution(
        self,
        results_dir,
        compute_backend="auto",
        max_working_memory_mb=1024,
        statement_batch_size=None,
        record_batch_size=None,
        initial_summary_mode="auto",
        max_initial_summary_items=100_000,
    ):
        self.params["results_dir"] = results_dir
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        self.compute_backend = compute_backend
        self.xp, self.active_backend = _resolve_array_backend(compute_backend)
        self.max_working_memory_mb = max_working_memory_mb
        self._max_working_memory_bytes = int(max_working_memory_mb * 1024 * 1024)
        self.statement_batch_size = statement_batch_size
        self.record_batch_size = record_batch_size
        self.initial_summary_mode = initial_summary_mode
        self.max_initial_summary_items = int(max_initial_summary_items)
        self.initial_summary_streamed = False
        print(f"Using {self.active_backend} compute backend.")

    def _ensure_memmap_dir(self):
        if self._memmap_dir is None:
            self._memmap_dir = tempfile.mkdtemp(
                prefix="fuzzylinguistics_",
                dir=self.params.get("results_dir", None),
            )
        return self._memmap_dir

    def _allocate_metric_array(self, name, shape):
        n_values = int(np.prod(shape))
        n_bytes = n_values * np.dtype(float).itemsize
        if n_bytes > self._max_working_memory_bytes:
            memmap_dir = self._ensure_memmap_dir()
            path = Path(memmap_dir) / f"{name}.dat"
            arr = np.memmap(path, dtype=float, mode="w+", shape=shape)
            arr[:] = 0.0
            return arr
        return np.zeros(shape)

    def _materialize_limit_bytes(self):
        return max(1024 * 1024, self._max_working_memory_bytes // 16)

    def _iter_slices(self, total, batch_size):
        for start in range(0, total, batch_size):
            yield start, min(start + batch_size, total)

    def _record_batch_size(self, n_records):
        if self.record_batch_size is not None:
            return max(1, min(int(self.record_batch_size), n_records))
        return n_records

    def _statement_batch_size(self, n_records, n_statements):
        if self.statement_batch_size is not None:
            return max(1, min(int(self.statement_batch_size), n_statements))
        bytes_per_statement = max(n_records, 1) * np.dtype(float).itemsize
        estimated = self._max_working_memory_bytes // max(bytes_per_statement * 8, 1)
        return max(1, min(n_statements, int(estimated)))

    def _paired_r_batch_size(self, n_records, p_batch_size, n_r_statements):
        bytes_per_pair = max(n_records, 1) * np.dtype(float).itemsize
        max_pairs = self._max_working_memory_bytes // max(bytes_per_pair * 4, 1)
        return max(1, min(n_r_statements, int(max_pairs // max(p_batch_size, 1)) or 1))

    def _compute_statement_metadata(self, statement_space, components):
        lengths = np.zeros(statement_space.total, dtype=int)
        weights = np.ones(statement_space.total, dtype=float)
        batch_size = max(1, min(statement_space.total, 100_000))

        for start, stop in self._iter_slices(statement_space.total, batch_size):
            indices = statement_space.decode_batch(start, stop)
            lengths[start:stop] = np.sum(indices > -1, axis=1)
            cur_weights = np.ones(stop - start, dtype=float)
            for k, component in enumerate(components):
                selected = indices[:, k]
                for j, weight in enumerate(component.operational_relevancy_weights):
                    mask = selected == j
                    if np.any(mask):
                        cur_weights[mask] = np.minimum(cur_weights[mask], weight)
            weights[start:stop] = cur_weights

        return lengths, weights

    def _should_store_statement_memberships(self, data_category):
        n_values = (
            self.n_summarizer_statements + self.n_qualifier_statements
        ) * data_category.n_records
        return n_values * np.dtype(float).itemsize <= self._max_working_memory_bytes // 2

    def _statement_membership_batch(self, data_category, statement_type, start, stop):
        if statement_type == "summarizer":
            cached = data_category.mu_p
            statement_space = self.summarizer_statement_indices
            components = self.summarizers
            base_memberships = data_category.mu_p_intermediate
        else:
            cached = data_category.mu_r
            statement_space = self.qualifier_statement_indices
            components = self.qualifiers
            base_memberships = data_category.mu_r_intermediate

        if cached is not None:
            return self.xp.asarray(cached[start:stop])

        indices = statement_space.decode_batch(start, stop)
        memberships = self.xp.ones(
            (stop - start, data_category.n_records),
            dtype=float,
        )

        for k, component in enumerate(components):
            selected = indices[:, k]
            for j in np.unique(selected[selected >= 0]):
                rows = np.where(selected == j)[0]
                if rows.size == 0:
                    continue
                base = self.xp.asarray(base_memberships[k, int(j), :])
                memberships[rows, :] = self.xp.minimum(memberships[rows, :], base)

        return memberships

    def _fill_statement_membership_cache(self, data_category, statement_type):
        if statement_type == "summarizer":
            total = self.n_summarizer_statements
            target = np.empty((total, data_category.n_records), dtype=float)
        else:
            total = self.n_qualifier_statements
            target = np.empty((total, data_category.n_records), dtype=float)

        batch_size = self._statement_batch_size(data_category.n_records, total)
        for start, stop in self._iter_slices(total, batch_size):
            memberships = self._statement_membership_batch(
                data_category,
                statement_type,
                start,
                stop,
            )
            target[start:stop, :] = _to_numpy(memberships)

        if statement_type == "summarizer":
            data_category.mu_p = target
        else:
            data_category.mu_r = target

    def _compute_statement_supports(self, data_category, statement_type):
        if statement_type == "summarizer":
            total = self.n_summarizer_statements
        else:
            total = self.n_qualifier_statements

        supports = np.zeros(total, dtype=float)
        batch_size = self._statement_batch_size(data_category.n_records, total)
        for start, stop in self._iter_slices(total, batch_size):
            memberships = self._statement_membership_batch(
                data_category,
                statement_type,
                start,
                stop,
            )
            supports[start:stop] = _to_numpy(self.xp.sum(memberships, axis=1))

        return supports

    def export_config(self, file_path: str):
        """
        Exports the current membership function configurations to a JSON file.

        This is useful for saving an auto-generated configuration for later use
        with `setup_fls_from_data`.

        Args:
            file_path: The path to the output JSON file.
        """

        def format_trapezoids(np_array: np.ndarray) -> list:
            """Converts a numpy array to a list and replaces NaN with None for JSON compatibility."""
            # Replace nan with a placeholder, convert to list, then replace placeholder with None
            return np.where(np.isnan(np_array), None, np_array).tolist()

        # --- 1. Extract and format summarizers ---
        summarizers_list = [
            {
                "dimension_name": s.attribute_name,
                "predicates": s.fuzzy_predicates,
                "trapezoidal_x_vals": format_trapezoids(s.trapezoidal_fuzzy_membership_function_x_vals),
                "relevancy_weights": s.operational_relevancy_weights,
            }
            for s in self.summarizers
        ]

        # --- 2. Extract and format qualifiers ---
        qualifiers_list = [
            {
                "dimension_name": q.attribute_name,
                "predicates": q.fuzzy_predicates,
                "trapezoidal_x_vals": format_trapezoids(q.trapezoidal_fuzzy_membership_function_x_vals),
                "relevancy_weights": q.operational_relevancy_weights,
            }
            for q in self.qualifiers
        ]

        # --- 3. Extract and format quantifiers ---
        if self.quantifiers:
            quantifiers_dict = {
                "dimension_name": "quantifier", # As per the original format
                "predicates": self.quantifiers.fuzzy_predicates,
                "trapezoidal_x_vals": format_trapezoids(self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals),
                "relevancy_weights": self.quantifiers.operational_relevancy_weights
            }
        else:
            quantifiers_dict = {}

        # --- 4. Combine into a single dictionary ---
        full_config = {
            "summarizers": summarizers_list,
            "qualifiers": qualifiers_list,
            "quantifiers": quantifiers_dict
        }

        # --- 5. Write to JSON file ---
        try:
            with open(file_path, "w") as f:
                json.dump(full_config, f, indent=4)
            print(f"Successfully exported membership function configuration to {file_path}")
        except Exception as e:
            print(f"Error exporting configuration to {file_path}: {e}")

    def compute_n_attributes(self):
        for data_category in self.data_categories:
            data_category.n_attributes = len(self.summarizers)
            if data_category.uses_qualifier:
                data_category.n_attributes += len(self.qualifiers)

        all_indexes = set()
        for data_category in self.data_categories:
            all_indexes.add(data_category.index)
        self.n_indexes = len(all_indexes)

    def generate_fls_two_models(self,
                               results_dir,
                               truth_threshold = 0.9,
                               first_model_name = None,
                               second_model_name = None,
                               compute_backend = "auto",
                               max_working_memory_mb = 1024,
                               statement_batch_size = None,
                               record_batch_size = None,
                               save_simplification_graph_gexf = False,
                               save_stage_graphs_gexf = False,
                               initial_summary_mode = "auto",
                               max_initial_summary_items = 100_000,
                               plot_membership_functions = False,
                               plot_membership_values = False,
                               save_initial_fls_txt = True,
                               save_initial_fls_csv = True,
                               save_initial_fls_latex = True,
                               simplify_fls = True):

        self._configure_execution(
            results_dir,
            compute_backend,
            max_working_memory_mb,
            statement_batch_size,
            record_batch_size,
            initial_summary_mode,
            max_initial_summary_items,
        )
        self.params["truth_threshold"] = truth_threshold

        if (first_model_name is None) or (second_model_name is None):
            model_names = [self.model_names[0], self.model_names[1]]
        else:
            model_names = [first_model_name, second_model_name]

        print(f"Model names = {model_names}")

        self.compute_n_attributes()

        if plot_membership_functions:
            self.plot_membership_functions()

        self.generate_statement_combinations()

        self.evaluate_fuzzy_memberships(model_names)

        if plot_membership_values:
            self.plot_membership_values(model_names)

        self.evaluate_truth_and_focus_values_two_models(model_names)

        self.aggregate_linguistic_qualities(model_names)

        self.sort_values(model_names)

        self.results = {
            "initial_linguistic_summary":{
                "linguistic_statements" : self.sorted_linguistic_statements,
                "truth_vals": self.sorted_truth_vals,
                "streamed": self.initial_summary_streamed,
                "count": getattr(self, "initial_summary_count", len(self.sorted_linguistic_statements)),
            }
        }

        comparison_str = model_names[0] + '_vs_' + model_names[1]

        if save_initial_fls_txt:
            self.save_linguistic_summary_txt(self.params["results_dir"] + '/' + comparison_str + '_initial_fuzzy_linguistic_summary.txt', False)

        if save_initial_fls_csv:
            csv_output_filename = self.params["results_dir"] + '/' + comparison_str + '_initial_fuzzy_linguistic_summary.csv'

            self.save_linguistic_summary_csv(csv_output_filename)

            if save_initial_fls_latex:
                self.csv_to_latex_table(csv_output_filename, csv_output_filename.replace(".csv", ".tex"))

        if simplify_fls:
            if save_simplification_graph_gexf or save_stage_graphs_gexf:
                self.generate_graph(model_names)
            if save_simplification_graph_gexf:
                self.export_graph(self.params["results_dir"] + '/linguistic_summary_graph', model_names)
            self.simplify_ls(
                self.params["truth_threshold"],
                self.params["results_dir"],
                model_names,
                save_stage_graphs_gexf=save_stage_graphs_gexf,
            )
            self.print_all_stats()

        return self.results

    def generate_fls_one_model(self,
                               results_dir,
                               truth_threshold = 0.9,
                               model_name = None,
                               compute_backend = "auto",
                               max_working_memory_mb = 1024,
                               statement_batch_size = None,
                               record_batch_size = None,
                               save_simplification_graph_gexf = False,
                               save_stage_graphs_gexf = False,
                               initial_summary_mode = "auto",
                               max_initial_summary_items = 100_000,
                               plot_membership_functions = False,
                               plot_membership_values = False,
                               save_initial_fls_txt = True,
                               save_initial_fls_csv = False,
                               save_initial_fls_latex = False,
                               simplify_fls = True):

        self._configure_execution(
            results_dir,
            compute_backend,
            max_working_memory_mb,
            statement_batch_size,
            record_batch_size,
            initial_summary_mode,
            max_initial_summary_items,
        )
        self.params["truth_threshold"] = truth_threshold

        if model_name is None:
            model_name = self.data_categories[0].model_name

        # Pre-compute total number of attributes for each category.
        self.compute_n_attributes()

        if plot_membership_functions:
            self.plot_membership_functions()

        self.generate_statement_combinations()

        self.evaluate_fuzzy_memberships([model_name])

        if plot_membership_values:
            self.plot_membership_values([model_name])

        self.evaluate_truth_and_focus_values([model_name])

        self.aggregate_linguistic_qualities([model_name])

        self.sort_values([model_name])

        self.results = {
            "initial_linguistic_summary":{
                "linguistic_statements" : self.sorted_linguistic_statements,
                "truth_vals": self.sorted_truth_vals,
                "streamed": self.initial_summary_streamed,
                "count": getattr(self, "initial_summary_count", len(self.sorted_linguistic_statements)),
            }
        }

        if save_initial_fls_txt:
            self.save_linguistic_summary_txt(self.params["results_dir"] + '/' + model_name + '_initial_fuzzy_linguistic_summary.txt', False)

        if save_initial_fls_csv:
            csv_output_filename = self.params["results_dir"] + '/' + model_name + '_initial_fuzzy_linguistic_summary.csv'

            self.save_linguistic_summary_csv(csv_output_filename)

            if save_initial_fls_latex:
                self.csv_to_latex_table(csv_output_filename, csv_output_filename.replace(".csv", ".tex"))

        if simplify_fls:
            if save_simplification_graph_gexf or save_stage_graphs_gexf:
                self.generate_graph([model_name])
            if save_simplification_graph_gexf:
                self.export_graph(self.params["results_dir"] + '/linguistic_summary_graph', [model_name])
            self.simplify_ls(
                self.params["truth_threshold"],
                self.params["results_dir"],
                [model_name],
                save_stage_graphs_gexf=save_stage_graphs_gexf,
            )
            self.print_all_stats()

        return self.results

    def generate_statement_combinations(self):
        # Generate list of list of possible index values.
        summarizer_counts = [
            len(summarizer.fuzzy_predicates)
            for summarizer in self.summarizers
        ]
        self.summarizer_statement_indices = StatementIndexSpace(
            summarizer_counts,
            self._materialize_limit_bytes(),
        )
        self.n_summarizer_statements = self.summarizer_statement_indices.shape[0]
        print("n_summarizer_statements = {}".format(self.n_summarizer_statements))

        qualifier_counts = [
            len(qualifier.fuzzy_predicates)
            for qualifier in self.qualifiers
        ]
        self.qualifier_statement_indices = StatementIndexSpace(
            qualifier_counts,
            self._materialize_limit_bytes(),
        )
        self.n_qualifier_statements = self.qualifier_statement_indices.shape[0]
        print("n_qualifier_statements = {}".format(self.n_qualifier_statements))

        self.n_statements = (
            self.n_summarizer_statements * self.n_qualifier_statements
        )
        print("n_statements = {}".format(self.n_statements))

    def evaluate_fuzzy_memberships(self, model_names):
        for data_category in self.data_categories:
            if data_category.model_name in model_names:
                print(f"Computing fuzzy memberships for the {data_category.category_name} for model {data_category.model_name}")

                # Compute Fuzzy memberships for every data point to every attribute+fuzzy_predicate pair that exists in P
                print("Computing base summarizer fuzzy membership values...")
                t0 = time.time()
                data_category.mu_p_intermediate = np.full((self.n_summarizers, self.max_summarizer_fuzzy_predicates, data_category.n_records), 1.0)
                for k in range(self.n_summarizers):
                    for j in range(len(self.summarizers[k].fuzzy_predicates)):
                        for start, stop in self._iter_slices(
                            data_category.n_records,
                            self._record_batch_size(data_category.n_records),
                        ):
                            x = self.xp.asarray(data_category.input_data[start:stop, k])
                            memberships = evaluate_trapezoidal_fuzzy_membership_backend(
                                self.summarizers[k].trapezoidal_fuzzy_membership_function_x_vals[j],
                                x,
                                self.xp,
                            )
                            data_category.mu_p_intermediate[k, j, start:stop] = _to_numpy(memberships)
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                # Compute summarizer statement metadata and fuzzy supports.
                print("Computing summarizer statement supports...")
                t0 = time.time()
                data_category.mu_p = None
                data_category.L_p, data_category.w_p = self._compute_statement_metadata(
                    self.summarizer_statement_indices,
                    self.summarizers,
                )
                if self._should_store_statement_memberships(data_category):
                    self._fill_statement_membership_cache(data_category, "summarizer")
                data_category.B_p = self._compute_statement_supports(
                    data_category,
                    "summarizer",
                )
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                # Compute Fuzzy memberships for every data point to every attribute+fuzzy_predicate pair that exists in R
                print("Computing base qualifier fuzzy membership values...")
                t0 = time.time()
                data_category.mu_r_intermediate = np.full((self.n_qualifiers, self.max_qualifier_fuzzy_predicates, data_category.n_records), 1.0)
                for k in range(self.n_qualifiers):
                    for j in range(len(self.qualifiers[k].fuzzy_predicates)):
                        for start, stop in self._iter_slices(
                            data_category.n_records,
                            self._record_batch_size(data_category.n_records),
                        ):
                            x = self.xp.asarray(data_category.output_data[start:stop, k])
                            memberships = evaluate_trapezoidal_fuzzy_membership_backend(
                                self.qualifiers[k].trapezoidal_fuzzy_membership_function_x_vals[j],
                                x,
                                self.xp,
                            )
                            data_category.mu_r_intermediate[k, j, start:stop] = _to_numpy(memberships)
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                # Compute qualifier statement metadata and fuzzy supports.
                print("Computing qualifier statement supports...")
                t0 = time.time()
                data_category.mu_r = None
                data_category.L_r, data_category.w_r = self._compute_statement_metadata(
                    self.qualifier_statement_indices,
                    self.qualifiers,
                )
                if self._should_store_statement_memberships(data_category):
                    self._fill_statement_membership_cache(data_category, "qualifier")
                data_category.B_r = self._compute_statement_supports(
                    data_category,
                    "qualifier",
                )
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def plot_membership_functions(self):
        n_plots = self.n_summarizers+self.n_qualifiers+1 # + 1 for the quantifiers

        fig, axs = plt.subplots(n_plots, 1, figsize=(3, n_plots), dpi=300)

        fnt_size_1 = 6
        fnt_size_2 = 6
        tick_size = 5
        line_width = 1

        for k in range(self.n_summarizers):
            n_fuzzy_predicates = len(self.summarizers[k].fuzzy_predicates)
            all_xs = []
            all_ys = []
            for j in range(n_fuzzy_predicates):
                xs = []
                ys = []

                xs.append(self.summarizers[k].trapezoidal_fuzzy_membership_function_x_vals[j][0])
                if j == 0:
                    ys.append(1)
                else:
                    ys.append(0)

                xs.append(self.summarizers[k].trapezoidal_fuzzy_membership_function_x_vals[j][1])
                ys.append(1)

                xs.append(self.summarizers[k].trapezoidal_fuzzy_membership_function_x_vals[j][2])
                ys.append(1)

                xs.append(self.summarizers[k].trapezoidal_fuzzy_membership_function_x_vals[j][3])
                if j == (n_fuzzy_predicates - 1):
                    ys.append(1)
                else:
                    ys.append(0)

                axs[k].plot(xs, ys, label=self.summarizers[k].fuzzy_predicates[j], linewidth=line_width)
                all_xs += xs
                all_ys += ys

            axs[k].set_xlabel(self.summarizers[k].display_name, fontsize=fnt_size_1)
            axs[k].set_ylabel('Membership Value', fontsize=fnt_size_1)
            axs[k].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fnt_size_2)

            scale_axis(axs[k], all_xs, all_ys, tick_size)

        for k in range(self.n_qualifiers):
            n_fuzzy_predicates = len(self.qualifiers[k].fuzzy_predicates)
            all_xs = []
            all_ys = []
            for j in range(len(self.qualifiers[k].fuzzy_predicates)):
                xs = []
                ys = []

                xs.append(self.qualifiers[k].trapezoidal_fuzzy_membership_function_x_vals[j][0])
                if j == 0:
                    ys.append(1)
                else:
                    ys.append(0)

                xs.append(self.qualifiers[k].trapezoidal_fuzzy_membership_function_x_vals[j][1])
                ys.append(1)

                xs.append(self.qualifiers[k].trapezoidal_fuzzy_membership_function_x_vals[j][2])
                ys.append(1)

                xs.append(self.qualifiers[k].trapezoidal_fuzzy_membership_function_x_vals[j][3])
                if j == (n_fuzzy_predicates - 1):
                    ys.append(1)
                else:
                    ys.append(0)

                axs[k+self.n_summarizers].plot(xs, ys, label=self.qualifiers[k].fuzzy_predicates[j], linewidth=line_width)
                all_xs += xs
                all_ys += ys

            axs[k+self.n_summarizers].set_xlabel(self.qualifiers[k].display_name, fontsize=fnt_size_1)
            axs[k+self.n_summarizers].set_ylabel('Membership Value', fontsize=fnt_size_1)
            axs[k+self.n_summarizers].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fnt_size_2)
            scale_axis(axs[k+self.n_summarizers], all_xs, all_ys, tick_size)


        n_fuzzy_predicates = len(self.quantifiers.fuzzy_predicates)
        all_xs = []
        all_ys = []
        for j in range(n_fuzzy_predicates):
            xs = []
            ys = []

            xs.append(self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals[j][0])
            if j == 0:
                ys.append(1)
            else:
                ys.append(0)

            xs.append(self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals[j][1])
            ys.append(1)

            xs.append(self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals[j][2])
            ys.append(1)

            xs.append(self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals[j][3])
            if j == (n_fuzzy_predicates - 1):
                ys.append(1)
            else:
                ys.append(0)

            axs[-1].plot(xs, ys, label=self.quantifiers.fuzzy_predicates[j], linewidth=line_width)
            all_xs += xs
            all_ys += ys

        axs[-1].set_xlabel("Quantifiers", fontsize=fnt_size_1)
        axs[-1].set_ylabel('Membership Value', fontsize=fnt_size_1)
        axs[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fnt_size_2)
        scale_axis(axs[-1], all_xs, all_ys, tick_size)

        plt.tight_layout()
        Path(self.params["results_dir"] + '/plots').mkdir(parents=True, exist_ok=True)
        plt.savefig(self.params["results_dir"] + '/plots/membership_functions.png', dpi=300)
        # plt.show()

    def plot_membership_values(self, model_names):
        tick_size = 5

        for data_category in self.data_categories:
            if data_category.model_name in model_names:
                fig, axs = plt.subplots(self.n_summarizers+self.n_qualifiers, 1, figsize=(12, 12))
                ax_id = 0
                for k in range(self.n_summarizers):
                    all_xs = []
                    all_ys = []
                    for j in range(len(self.summarizers[k].fuzzy_predicates)):
                        xs = data_category.input_data[:, k].tolist()
                        ys = data_category.mu_p_intermediate[k, j, :].tolist()

                        axs[ax_id].scatter(xs, ys, label=self.summarizers[k].fuzzy_predicates[j], alpha=1/(self.n_summarizers+1))
                        all_xs += xs
                        all_ys += ys

                    axs[ax_id].set_xlabel(self.summarizers[k].display_name)
                    axs[ax_id].set_ylabel('Fuzzy Membership Value')
                    axs[ax_id].legend()
                    scale_axis(axs[ax_id], all_xs, all_ys, tick_size)
                    ax_id = ax_id + 1

                for k in range(self.n_qualifiers):
                    all_xs = []
                    all_ys = []
                    for j in range(len(self.qualifiers[k].fuzzy_predicates)):
                        xs = data_category.output_data[:, k].tolist()
                        ys = data_category.mu_r_intermediate[k, j, :].tolist()

                        axs[ax_id].scatter(xs, ys, label=self.qualifiers[k].fuzzy_predicates[j], alpha=1/(self.n_qualifiers+1))
                        all_xs += xs
                        all_ys += ys

                    axs[ax_id].set_xlabel(self.qualifiers[k].display_name)
                    axs[ax_id].set_ylabel('Fuzzy Membership Value')
                    axs[ax_id].legend()
                    scale_axis(axs[ax_id], all_xs, all_ys, tick_size)
                    ax_id = ax_id + 1

                plt.tight_layout()
                Path(self.params["results_dir"] + '/plots').mkdir(parents=True, exist_ok=True)
                plt.savefig(self.params["results_dir"] + '/plots/' + data_category.model_name + '_' + data_category.category_name + '_membership_values.png', dpi=300)
                # plt.show()

    def evaluate_truth_and_focus_values(self, model_names):
        print(f"Computing truth and focus values...")
        t0 = time.time()

        metric_shape = (
            self.n_indexes,
            self.n_quantifier_fuzzy_predicates,
            self.n_qualifier_statements,
            self.n_summarizer_statements,
        )
        self.truth_vals = self._allocate_metric_array("truth_vals", metric_shape)
        self.focus_vals = self._allocate_metric_array("focus_vals", metric_shape)
        self.simplicity_vals = self._allocate_metric_array("simplicity_vals", metric_shape)
        self.operational_relevancy_vals = self._allocate_metric_array(
            "operational_relevancy_vals",
            metric_shape,
        )

        total_records = 0
        for data_category in self.data_categories:
            if data_category.model_name in model_names:
                total_records += data_category.n_records

        for data_category in self.data_categories:
            if data_category.model_name in model_names:
                p_batch_size = self._statement_batch_size(
                    data_category.n_records,
                    self.n_summarizer_statements,
                )
                for p_start, p_stop in self._iter_slices(
                    self.n_summarizer_statements,
                    p_batch_size,
                ):
                    mu_p = self._statement_membership_batch(
                        data_category,
                        "summarizer",
                        p_start,
                        p_stop,
                    )
                    p_indices = np.arange(p_start, p_stop)[None, :]
                    b_vals = data_category.B_p[p_start:p_stop][None, :]
                    r_batch_size = self._paired_r_batch_size(
                        data_category.n_records,
                        p_stop - p_start,
                        self.n_qualifier_statements,
                    )

                    for r_start, r_stop in self._iter_slices(
                        self.n_qualifier_statements,
                        r_batch_size,
                    ):
                        mu_r = self._statement_membership_batch(
                            data_category,
                            "qualifier",
                            r_start,
                            r_stop,
                        )
                        a_vals = _to_numpy(
                            self.xp.sum(
                                self.xp.minimum(mu_r[:, None, :], mu_p[None, :, :]),
                                axis=2,
                            )
                        )
                        r_indices = np.arange(r_start, r_stop)[:, None]
                        c_vals = np.zeros_like(a_vals, dtype=float)
                        nonzero = a_vals != 0

                        if self.params["LANGUAGE_MODE"] == 0:
                            total_b = np.zeros_like(b_vals)
                            for data_category_2 in self.data_categories:
                                if data_category_2.model_name in model_names:
                                    total_b += data_category_2.B_p[p_start:p_stop][None, :]
                            b_for_focus = total_b
                            unconditional = (p_indices == 0) | (r_indices == 0)
                            c_vals[nonzero & unconditional] = (
                                a_vals[nonzero & unconditional] / total_records
                            )
                            conditional = nonzero & ~unconditional & (total_b > 0)
                            c_vals[conditional] = a_vals[conditional] / np.broadcast_to(total_b, a_vals.shape)[conditional]
                        elif self.params["LANGUAGE_MODE"] == 1:
                            b_for_focus = b_vals
                            if data_category.uses_qualifier:
                                mask = nonzero & (p_indices == 0) & (r_indices == 0)
                                c_vals[mask] = -0.5
                                mask = nonzero & (p_indices == 0) & (r_indices > 0)
                                c_vals[mask] = a_vals[mask] / data_category.n_records
                                mask = nonzero & (p_indices > 0) & (r_indices == 0)
                                c_vals[mask] = np.broadcast_to(b_vals, a_vals.shape)[mask] / data_category.n_records
                                mask = nonzero & (p_indices > 0) & (r_indices > 0) & (b_vals > 0)
                                c_vals[mask] = a_vals[mask] / np.broadcast_to(b_vals, a_vals.shape)[mask]
                            else:
                                mask = nonzero & (p_indices == 0)
                                c_vals[mask] = a_vals[mask] / total_records
                                mask = nonzero & (p_indices > 0) & (b_vals > 0)
                                c_vals[mask] = a_vals[mask] / np.broadcast_to(b_vals, a_vals.shape)[mask]
                        else:
                            raise(ValueError(f"Error unrecognized self.params['LANGUAGE_MODE'] = {self.params['LANGUAGE_MODE']}"))

                        simplicity = 1.0 - (
                            (
                                data_category.L_p[p_start:p_stop][None, :]
                                + data_category.L_r[r_start:r_stop][:, None]
                            )
                            / data_category.n_attributes
                        )

                        for q in range(self.n_quantifier_fuzzy_predicates):
                            truth = evaluate_trapezoidal_fuzzy_membership(
                                self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals[q],
                                c_vals,
                            )
                            q_weight = self.quantifiers.operational_relevancy_weights[q]
                            relevance = np.minimum(
                                q_weight,
                                np.minimum(
                                    data_category.w_r[r_start:r_stop][:, None],
                                    data_category.w_p[p_start:p_stop][None, :],
                                ),
                            )

                            self.truth_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = truth
                            self.focus_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = truth * (
                                np.broadcast_to(b_for_focus, a_vals.shape)
                                / total_records
                            )
                            self.simplicity_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = simplicity
                            self.operational_relevancy_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = relevance

        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def evaluate_truth_and_focus_values_two_models(self, model_names):
        print(f"Computing truth and focus values...")
        t0 = time.time()

        metric_shape = (
            self.n_indexes,
            self.n_quantifier_fuzzy_predicates,
            self.n_qualifier_statements,
            self.n_summarizer_statements,
        )
        self.truth_vals = self._allocate_metric_array("truth_vals", metric_shape)
        self.focus_vals = self._allocate_metric_array("focus_vals", metric_shape)
        self.simplicity_vals = self._allocate_metric_array("simplicity_vals", metric_shape)
        self.operational_relevancy_vals = self._allocate_metric_array(
            "operational_relevancy_vals",
            metric_shape,
        )

        n_records = {}
        total_records_1 = 0
        total_records_2 = 0
        category_model_data = {}
        for data_category in self.data_categories:
            key = data_category.category_name + "|" + data_category.model_name
            category_model_data[key] = data_category
            if key not in n_records:
                n_records[key] = data_category.n_records
            else:
                n_records[key] += data_category.n_records
            if data_category.model_name == model_names[0]:
                total_records_1 += data_category.n_records
            if data_category.model_name == model_names[1]:
                total_records_2 += data_category.n_records

        max_total_records = max(total_records_1, total_records_2)

        for data_category in self.data_categories:
            if data_category.model_name == model_names[0]:
                key_1 = data_category.category_name + "|" + model_names[0]
                key_2 = data_category.category_name + "|" + model_names[1]
                if key_2 not in category_model_data:
                    raise KeyError(f"Missing comparison dataset for {key_2}")
                data_category_2 = category_model_data[key_2]

                p_batch_size = self._statement_batch_size(
                    max(data_category.n_records, data_category_2.n_records),
                    self.n_summarizer_statements,
                )
                for p_start, p_stop in self._iter_slices(
                    self.n_summarizer_statements,
                    p_batch_size,
                ):
                    mu_p_1 = self._statement_membership_batch(
                        data_category,
                        "summarizer",
                        p_start,
                        p_stop,
                    )
                    mu_p_2 = self._statement_membership_batch(
                        data_category_2,
                        "summarizer",
                        p_start,
                        p_stop,
                    )
                    p_indices = np.arange(p_start, p_stop)[None, :]
                    b_1 = data_category.B_p[p_start:p_stop][None, :]
                    b_2 = data_category_2.B_p[p_start:p_stop][None, :]

                    if self.params["LANGUAGE_MODE"] == 0:
                        b_vals = np.zeros_like(b_1)
                        for data_category_iter in self.data_categories:
                            key_3 = data_category_iter.category_name + "|" + model_names[0]
                            key_4 = data_category_iter.category_name + "|" + model_names[1]
                            if key_3 in category_model_data and key_4 in category_model_data:
                                b_vals += np.maximum(
                                    category_model_data[key_3].B_p[p_start:p_stop][None, :],
                                    category_model_data[key_4].B_p[p_start:p_stop][None, :],
                                )
                    elif self.params["LANGUAGE_MODE"] == 1:
                        b_vals = np.maximum(b_1, b_2)
                    else:
                        raise(ValueError(f"Error unrecognized self.params['LANGUAGE_MODE'] = {self.params['LANGUAGE_MODE']}"))

                    r_batch_size = self._paired_r_batch_size(
                        max(data_category.n_records, data_category_2.n_records),
                        p_stop - p_start,
                        self.n_qualifier_statements,
                    )
                    for r_start, r_stop in self._iter_slices(
                        self.n_qualifier_statements,
                        r_batch_size,
                    ):
                        mu_r_1 = self._statement_membership_batch(
                            data_category,
                            "qualifier",
                            r_start,
                            r_stop,
                        )
                        mu_r_2 = self._statement_membership_batch(
                            data_category_2,
                            "qualifier",
                            r_start,
                            r_stop,
                        )
                        a_1 = _to_numpy(
                            self.xp.sum(
                                self.xp.minimum(mu_r_1[:, None, :], mu_p_1[None, :, :]),
                                axis=2,
                            )
                        )
                        a_2 = _to_numpy(
                            self.xp.sum(
                                self.xp.minimum(mu_r_2[:, None, :], mu_p_2[None, :, :]),
                                axis=2,
                            )
                        )
                        a_vals = a_1 - a_2
                        c_vals = np.zeros_like(a_vals, dtype=float)
                        r_indices = np.arange(r_start, r_stop)[:, None]
                        nonzero = a_vals != 0

                        if self.params["LANGUAGE_MODE"] == 0:
                            mask = nonzero & (p_indices == 0)
                            c_vals[mask] = a_vals[mask] / max_total_records
                            mask = nonzero & (p_indices > 0) & (b_vals > 0)
                            c_vals[mask] = a_vals[mask] / np.broadcast_to(b_vals, a_vals.shape)[mask]
                        elif self.params["LANGUAGE_MODE"] == 1:
                            if data_category.uses_qualifier:
                                mask = nonzero & (p_indices == 0)
                                c_vals[mask] = a_vals[mask] / max(n_records[key_1], n_records[key_2])
                                mask = nonzero & (p_indices > 0) & (b_vals > 0)
                                c_vals[mask] = a_vals[mask] / np.broadcast_to(b_vals, a_vals.shape)[mask]
                            else:
                                mask = nonzero & (p_indices == 0)
                                c_vals[mask] = a_vals[mask] / max_total_records
                                mask = nonzero & (p_indices > 0) & (b_vals > 0)
                                c_vals[mask] = a_vals[mask] / np.broadcast_to(b_vals, a_vals.shape)[mask]

                        no_support = (a_vals == 0) & (a_1 == 0) & (a_2 == 0)
                        c_vals[no_support] = float('nan')

                        focus_support = (b_1 + b_2) / (n_records[key_1] + n_records[key_2])
                        simplicity = 1.0 - (
                            (
                                data_category.L_p[p_start:p_stop][None, :]
                                + data_category.L_r[r_start:r_stop][:, None]
                            )
                            / data_category.n_attributes
                        )

                        for q in range(self.n_quantifier_fuzzy_predicates):
                            truth = evaluate_trapezoidal_fuzzy_membership(
                                self.quantifiers.trapezoidal_fuzzy_membership_function_x_vals[q],
                                c_vals,
                            )
                            q_weight = self.quantifiers.operational_relevancy_weights[q]
                            relevance = np.minimum(
                                q_weight,
                                np.minimum(
                                    data_category.w_r[r_start:r_stop][:, None],
                                    data_category.w_p[p_start:p_stop][None, :],
                                ),
                            )

                            self.truth_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = truth
                            self.focus_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = truth * np.broadcast_to(focus_support, a_vals.shape)
                            self.simplicity_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = simplicity
                            self.operational_relevancy_vals[
                                data_category.index,
                                q,
                                r_start:r_stop,
                                p_start:p_stop,
                            ] = relevance


        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def aggregate_linguistic_qualities(self, model_names):
        self.aggregated_values = {}
        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                shape = (
                    self.n_quantifier_fuzzy_predicates,
                    self.n_qualifier_statements,
                    self.n_summarizer_statements,
                )
                aggregated = self._allocate_metric_array(
                    f"aggregated_values_{data_category.index}",
                    shape,
                )
                p_batch_size = self._statement_batch_size(
                    data_category.n_records,
                    self.n_summarizer_statements,
                )
                for p_start, p_stop in self._iter_slices(
                    self.n_summarizer_statements,
                    p_batch_size,
                ):
                    quality = (
                        1.0
                        + self.w_focus * self.focus_vals[data_category.index, :, :, p_start:p_stop]
                        + self.w_complexity * self.simplicity_vals[data_category.index, :, :, p_start:p_stop]
                    )
                    aggregated[:, :, p_start:p_stop] = (
                        self.operational_relevancy_vals[data_category.index, :, :, p_start:p_stop]
                        * self.truth_vals[data_category.index, :, :, p_start:p_stop]
                        * quality
                    ) / (1 + self.w_focus + self.w_complexity)
                self.aggregated_values[data_category.index] = aggregated

    def sort_values(self, model_names):
        print("Sorting values...")
        t0 = time.time()

        total_items = 0
        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                n_r = self.n_qualifier_statements if data_category.uses_qualifier else 1
                total_items += (
                    self.n_quantifier_fuzzy_predicates
                    * n_r
                    * self.n_summarizer_statements
                )

        stream_large_summary = (
            self.initial_summary_mode == "stream"
            or (
                self.initial_summary_mode == "auto"
                and total_items > self.max_initial_summary_items
            )
        )

        if stream_large_summary:
            self.initial_summary_streamed = True
            self.initial_summary_count = total_items
            self.sorted_linguistic_statements = np.array([], dtype=object)
            self.sorted_aggregated_values = np.array([], dtype=float)
            self.sorted_truth_vals = np.array([], dtype=float)
            self.sorted_focus_vals = np.array([], dtype=float)
            self.sorted_operational_vals = np.array([], dtype=float)
            self.sorted_simplity_vals = np.array([], dtype=float)
            t1 = time.time()
            print(
                "Initial summary has {} statements; skipped in-memory "
                "rendering in '{}' mode. Done. Took {}s.".format(
                    total_items,
                    self.initial_summary_mode,
                    round(t1 - t0, 3),
                )
            )
            return

        self.linguistic_statements = []
        aggregated_values = []
        all_truth_vals = []
        all_operational_vals = []
        all_focus_vals = []
        all_simplicity_vals = []

        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                for q in range(self.n_quantifier_fuzzy_predicates):
                    for r in range(self.n_qualifier_statements):
                        for p in range(self.n_summarizer_statements):
                            if data_category.uses_qualifier or ((not data_category.uses_qualifier) and (r == 0)):
                                self.linguistic_statements.append(self.get_linguistic_statement(data_category, [q], [r], [p], False, model_names))
                                aggregated_values.append(self.aggregated_values[data_category.index][q, r, p])
                                all_truth_vals.append(self.truth_vals[data_category.index, q, r, p])
                                all_operational_vals.append(self.operational_relevancy_vals[data_category.index, q, r, p])
                                all_focus_vals.append(self.focus_vals[data_category.index, q, r, p])
                                all_simplicity_vals.append(self.simplicity_vals[data_category.index, q, r, p])

        inds = (-np.array(aggregated_values)).argsort()
        inds = inds.astype(int)
        self.linguistic_statements = np.array(self.linguistic_statements)
        aggregated_values = np.array(aggregated_values)
        all_truth_vals = np.array(all_truth_vals)
        all_operational_vals = np.array(all_operational_vals)
        all_focus_vals = np.array(all_focus_vals)
        all_simplicity_vals = np.array(all_simplicity_vals)
        self.sorted_linguistic_statements = self.linguistic_statements[inds]
        self.sorted_aggregated_values = aggregated_values[inds]
        self.sorted_truth_vals = all_truth_vals[inds]
        self.sorted_focus_vals = all_focus_vals[inds]
        self.sorted_operational_vals = all_operational_vals[inds]
        self.sorted_simplity_vals = all_simplicity_vals[inds]

        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def save_linguistic_summary_txt(self, output_filename, save_nonzero_agg_values_only = False):
        print("Saving linguitic summary to txt file...")
        t0 = time.time()
        file = open(output_filename, "w")
        if getattr(self, "initial_summary_streamed", False):
            file.write(
                "Initial linguistic summary was not rendered in memory because "
                f"it contains {self.initial_summary_count} statements. "
                "Use a larger max_initial_summary_items value or "
                "initial_summary_mode='memory' to force full rendering.\n"
            )
            file.close()
            print("Saved linguistic txt summary metadata: " + output_filename)
            t1 = time.time()
            print("Done. Took {}s.".format(round(t1 - t0, 3)))
            return
        for i in range(len(self.sorted_linguistic_statements)):
            if ((save_nonzero_agg_values_only == True) and (self.sorted_aggregated_values[i] > 0)) or (save_nonzero_agg_values_only == False):
                file.write(f"{round(self.sorted_aggregated_values[i], 3)} | {self.sorted_linguistic_statements[i]}\n")
        file.close()
        print("Saved linguistic txt summary: " + output_filename)
        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def save_linguistic_summary_csv(self, output_filename):
        print("Saving linguitic summary to csv file...")
        t0 = time.time()
        file = open(output_filename, "w")
        file.write("Rank, V(S), O(S), T(S), F(S), 1-C(S), S\n")
        if getattr(self, "initial_summary_streamed", False):
            file.write(
                "NA, NA, NA, NA, NA, NA, "
                f"Initial linguistic summary contains {self.initial_summary_count} "
                "statements and was not rendered in memory.\n"
            )
            file.close()
            print("Saved linguistic csv summary metadata: " + output_filename)
            t1 = time.time()
            print("Done. Took {}s.".format(round(t1 - t0, 3)))
            return
        for i in range(len(self.sorted_linguistic_statements)):
            v = round(self.sorted_aggregated_values[i], 3)
            o = round(self.sorted_operational_vals[i], 3)
            t = round(self.sorted_truth_vals[i], 3)
            f = round(self.sorted_focus_vals[i], 3)
            c = round(self.sorted_simplity_vals[i], 3)
            s = self.sorted_linguistic_statements[i]
            file.write("{}, {}, {}, {}, {}, {}, {}\n".format(i+1, v, o, t, f, c, s))
        file.close()
        print("Saved linguistic csv summary: " + output_filename)
        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def csv_to_latex_table(self, file_name, output_file_name):
        with open(file_name, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data = [row for row in reader]

        with open(output_file_name, 'w') as f:
            f.write('\\begin{table}\n')
            f.write('\t\\centering\n')
            f.write('\t\\begin{tabularx}{\\columnwidth}{*{6}{>{\\centering\\arraybackslash}X}}\n')
            f.write('\t\t\\hline\n')

            # Write header
            cur_str = '\t\t' + ' & '.join(data[0][:-1]) + ' \\\\\n'
            f.write(cur_str.replace(',', ' &'))
            f.write('\t\t\\hline\n')

            for row in data[1:]:
                # Write main row
                f.write('\t\t' + ' & '.join(row[:-1]) + ' \\\\\n')
                # Write row with S column
                cur_str = '\t\t& \\multicolumn{4}{>{\\hsize=\\dimexpr5\\hsize+8\\tabcolsep\\relax}X}{' + row[-1] + '} \\\\\n'
                f.write(cur_str.replace(',', ' &'))
                f.write('\t\t\\hline\n')

            f.write('\t\\end{tabularx}\n')
            f.write('\\end{table}\n')

        print("Saved {}".format(output_file_name))

    def get_node_code(self, q,r,p):
        node_code = []
        node_code.append(q)
        for k in range(len(self.qualifiers)):
            node_code.append(self.qualifier_statement_indices[r][k])
        for k in range(len(self.summarizers)):
            node_code.append(self.summarizer_statement_indices[p][k])

        return node_code

    def get_node_code_str(self, q,r,p):
        node_code = ''
        node_code += str(q)
        for k in range(self.n_qualifiers):
            node_code += str(self.qualifier_statement_indices[r][k])
        for k in range(self.n_summarizers):
            node_code += str(self.summarizer_statement_indices[p][k])

        return node_code

    def get_list_of_all_node_codes(self):
        all_node_codes = {}
        for q, r, p in product(range(self.n_quantifier_fuzzy_predicates), range(self.n_qualifier_statements), range(self.n_summarizer_statements)):
            all_node_codes[tuple([q, r, p])] = tuple(self.get_node_code(q, r, p))

        return all_node_codes

    def get_children_from_parent(self, parent_node_code):
        # Convert to a mutable list only once
        parent_list = list(parent_node_code)
        children = []

        for i, val in enumerate(parent_list):
            if val == -1:
                if i <= self.n_qualifiers:
                    # Precompute fuzzy_len once per index where val == -1
                    n_fuzzy_predicates = len(self.qualifiers[i-1].fuzzy_predicates)
                else:
                    # Precompute fuzzy_len once per index where val == -1
                    n_fuzzy_predicates = len(self.summarizers[i-self.n_qualifiers-1].fuzzy_predicates)

                for j in range(n_fuzzy_predicates):
                    parent_list[i] = j
                    children.append(tuple(parent_list[:]))

                # Revert to the original value
                parent_list[i] = -1

        return children

    def build_child_map(self, all_node_codes):
        return [self.get_children_from_parent(all_node_codes[key]) for key in all_node_codes]

    def generate_graph(self, model_names):
        print("Generating graph...")
        t0 = time.time()

        print("Generating list of all node codes...")
        all_node_codes = self.get_list_of_all_node_codes()
        reversed_all_node_codes = {v: k for k, v in all_node_codes.items()}
        print("Building child map...")
        parent_to_children = self.build_child_map(all_node_codes)

        self.G = []
        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                self._scale_aggregated_for_simplification(data_category.index)

                V = []
                E = []

                colors = []
                labels = []
                sizes = []
                data = []

                # Add root node first.
                colors.append([0.5, 0.5, 0.5])
                root_node_id = 'Linguistic Summaries'
                labels.append(root_node_id)
                sizes.append(100.0)
                data.append({
                                'data_type': data_category.index,
                                'q': -1,
                                'r': -1,
                                'p': -1,
                                'v_s': -1,
                                't_s': -1,
                                'f_s': -1,
                                'o_s': -1,
                                'c_s': -1,
                                'T': False,
                                'R': False,
                                'P': True
                            })
                V.append(root_node_id)
                E.append([])

                # Calculate the total number of iterations
                total_iterations = (
                    self.n_quantifier_fuzzy_predicates *
                    self.n_qualifier_statements *
                    self.n_summarizer_statements
                )

                # c_offset = np.random.random()
                c_offset = 0.5
                indx = -1
                for q, r, p in tqdm(
                        product(
                            range(self.n_quantifier_fuzzy_predicates),
                            range(self.n_qualifier_statements),
                            range(self.n_summarizer_statements)
                        ),
                        total=total_iterations,
                        desc='Processing'
                    ):
                        indx += 1
                        if (not data_category.uses_qualifier) and (r > 0):
                            continue

                        parent_s = self.get_linguistic_statement(data_category, [q], [r], [p], False, model_names)
                        V.append(parent_s)
                        E.append([])

                        hue = ((data_category.L_p[p]+data_category.L_r[r]) / (data_category.n_attributes+1)) * 0.75 + c_offset
                        if hue > 1:
                            hue = hue - 1
                        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
                        colors.append(rgb)
                        labels.append(parent_s)
                        # labels.append(self.get_node_code_str(q,r,p))
                        sizes.append(self.aggregated_values[data_category.index][q, r, p]*100)
                        data.append({
                            'data_type': data_category.index,
                            'q': q,
                            'r': r,
                            'p': p,
                            'v_s': self.aggregated_values[data_category.index][q, r, p],
                            't_s': self.truth_vals[data_category.index][q, r, p],
                            'f_s': self.focus_vals[data_category.index][q, r, p],
                            'o_s': self.operational_relevancy_vals[data_category.index][q, r, p],
                            'c_s': self.simplicity_vals[data_category.index][q, r, p],
                            'T': None,
                            'R': None,
                            'P': False
                        })

                        # If no summarizer and no qualifier then add to the root node placeholder.
                        if (p == 0) and (r == 0):
                            E[0].append(parent_s)

                        children = parent_to_children[indx]
                        for child in children:
                            child_qrp = reversed_all_node_codes[child]
                            child_s = self.get_linguistic_statement(data_category, [child_qrp[0]], [child_qrp[1]], [child_qrp[2]], False, model_names)
                            E[-1].append(child_s)

                self.G.append({
                    'nodes': V,
                    'edges': E,
                    'colors': colors,
                    'labels': labels,
                    'sizes': sizes,
                    'data': data
                })

        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def convert_to_networkx(self, data_index):
        G = nx.DiGraph()

        # Add nodes with attributes
        for i, node_id in enumerate(self.G[data_index]['nodes']):
            color = self.G[data_index]['colors'][i]
            label = self.G[data_index]['labels'][i]
            size = self.G[data_index]['sizes'][i]
            r, g, b = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            data = self.G[data_index]['data'][i]

            # Add node with placeholder positions (e.g., computed later)
            G.add_node(
                node_id,
                label=label,
                size=size,
                viz={
                    'color': {'r': r, 'g': g, 'b': b},
                    'size': size,
                    'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}  # Default position
                },
                data = data
            )

        # Add edges
        for i, neighbors in enumerate(self.G[data_index]['edges']):
            for neighbor in neighbors:
                G.add_edge(self.G[data_index]['nodes'][i], neighbor)

        return G

    def construct_qualifiers(self, rs):
        qualifiers = []
        for r in rs:
            qualifier = ""
            for k in range(len(self.qualifiers)):
                j = self.qualifier_statement_indices[r][k]
                if j >= 0:
                    attribute = self.qualifiers[k].attribute_name

                    fuzzy_predicate = self.qualifiers[k].fuzzy_predicates[j]
                    if qualifier != "":
                        qualifier = qualifier + " and " + fuzzy_predicate + " " + attribute
                    else:
                        qualifier = qualifier + fuzzy_predicate + " " + attribute
            qualifiers.append(qualifier)

        return qualifiers

    def construct_summarizers(self, ps):
        summarizers = []
        for p in ps:
            summarizer = ""
            for k in range(len(self.summarizers)):
                j = self.summarizer_statement_indices[p][k]
                if j >= 0:
                    attribute = self.summarizers[k].attribute_name

                    fuzzy_predicate = self.summarizers[k].fuzzy_predicates[j]
                    if summarizer != "":
                        summarizer = summarizer + " and " + fuzzy_predicate + " " + attribute
                    else:
                        summarizer = summarizer + fuzzy_predicate + " " + attribute
            summarizers.append(summarizer)

        return summarizers

    def get_linguistic_statement(self, data_category, qs, rs, ps, negative_flag, model_names):
        sorted_lists = sorted(zip(ps, rs, qs))
        ps, rs, qs = map(list, zip(*sorted_lists))

        # Construct quantifier
        quantifiers = []
        for q in qs:
            quantifiers.append(self.quantifiers.fuzzy_predicates[q])
        all_quantifiers_equal = all(item == quantifiers[0] for item in quantifiers)

        # Construct qualifier
        qualifiers = self.construct_qualifiers(rs)
        all_qualifiers_equal = all(item == qualifiers[0] for item in qualifiers)

        # Construct summarizer
        summarizers = self.construct_summarizers(ps)
        all_summarizers_equal = all(item == summarizers[0] for item in summarizers)

        if negative_flag:
            neg_s = "except "
        else:
            neg_s = ""

        data_name = 'data'
        preposition = "with "
        qual_sum_conj = 'are '

        if len(model_names) == 1:
            comparison_suffix = ""
        elif len(model_names) == 2:
            if quantifiers[0].lower() == "none":
                comparison_suffix = f"from {model_names[0]} or from {model_names[1]}"
            elif quantifiers[0].lower() == "the same amount":
                comparison_suffix = f"from {model_names[0]} and from {model_names[1]}"
            else:
                comparison_suffix = f"from {model_names[0]} then from {model_names[1]}"
        else:
            raise(ValueError("Error: More than 2 models not currently supported."))

        # elif self.params["LANGUAGE_MODE"] == 1:
        if summarizers[0] == "" and qualifiers[0] == "":
            S = f"{quantifiers[0]} of the data {qual_sum_conj} {data_category.category_name}"
        elif summarizers[0] == "":
            S = f"{quantifiers[0]} of the {data_category.category_name} {qual_sum_conj} {qualifiers[0]}"
        elif qualifiers[0] == "":
            S = f"{quantifiers[0]} of the {data_category.category_name} {qual_sum_conj} {summarizers[0]}"
        else:
            S = f"Of the {data_category.category_name} {neg_s}{preposition}{summarizers[0]} {quantifiers[0]} {qual_sum_conj} {qualifiers[0]}"

        if len(ps) > 1:
            if all_quantifiers_equal == False:
                print("Error constructing compound statement:")
                for i in range(len(qs)):
                    print(self.get_linguistic_statement(data_category, [qs[i]], [rs[i]], [ps[i]], negative_flag, model_names))
                raise(ValueError("Error: All quantifiers should be equal when trying to get a compound statement."))

            if all_summarizers_equal and all_qualifiers_equal:
                raise(ValueError("Error: All summarizers, quantifiers, and qualifiers were equal when attempting to construct a compound statement."))

            for i in range(1, len(ps)):
                if summarizers[i] == "" and qualifiers[i] == "":
                    raise(ValueError("Error: Summarizer and qualifier should not be empty when attempting to construct a compound statement."))

                if all_qualifiers_equal:
                    S += f" or {summarizers[i]}"

                if all_summarizers_equal:
                    S += f" or {qualifiers[i]}"

                if (all_qualifiers_equal == False) and (all_summarizers_equal == False):
                    raise(ValueError(f"Error: Either all qualifiers or all summarizer should be equal when attempting to construct a compount statement."))

        S += f" {comparison_suffix}."

        S = S[0].upper() + S[1:].lower()
        S = re.sub(r'\s{2,}', ' ', S)
        S = S.replace("_", ' ').replace(" .", ".")


        # if len(ps) == 1:
            # S = str(self.get_node_code(qs[0], rs[0], ps[0])) + S

        # if all_quantifiers_equal:
        #     S = "all_quantiifers_equal | " + S
        # if all_qualifiers_equal:
        #     S = "all_summarizers_equal | " + S
        # if all_qualifiers_equal:
        #     S = "all_qualifiers_equal | " + S

        return S

    def _scale_aggregated_for_simplification(self, data_index):
        if data_index not in self._aggregated_scaled_indexes:
            self.aggregated_values[data_index] = scale_array_to_01(
                self.aggregated_values[data_index]
            )
            self._aggregated_scaled_indexes.add(data_index)

    def _iter_valid_statement_triples(self, data_category):
        for q, r, p in product(
            range(self.n_quantifier_fuzzy_predicates),
            range(self.n_qualifier_statements),
            range(self.n_summarizer_statements),
        ):
            if (not data_category.uses_qualifier) and (r > 0):
                continue
            yield q, r, p

    def _node_code_to_qrp(self, node_code):
        q = node_code[0]
        r_values = node_code[1:1 + self.n_qualifiers]
        p_values = node_code[1 + self.n_qualifiers:]
        r = self.qualifier_statement_indices.encode(r_values)
        p = self.summarizer_statement_indices.encode(p_values)
        return q, r, p

    def _node_id_from_code(self, data_category, node_code, model_names):
        q, r, p = self._node_code_to_qrp(node_code)
        return self.get_linguistic_statement(
            data_category,
            [q],
            [r],
            [p],
            False,
            model_names,
        )

    def _has_true_ancestor(self, data_category, node_code, true_nodes, model_names):
        active_positions = [
            i for i in range(1, len(node_code))
            if node_code[i] >= 0
        ]
        for n_positions in range(1, len(active_positions) + 1):
            for positions in itertools.combinations(active_positions, n_positions):
                ancestor = list(node_code)
                for position in positions:
                    ancestor[position] = -1
                ancestor_id = self._node_id_from_code(
                    data_category,
                    tuple(ancestor),
                    model_names,
                )
                if ancestor_id in true_nodes:
                    return True
        return False

    def _close_sibling_node_ids(self, data_category, node_code, node_data, model_names):
        siblings = set()
        for i in range(1, len(node_code)):
            if node_code[i] < 0:
                continue
            if i <= self.n_qualifiers:
                n_fuzzy_predicates = len(self.qualifiers[i - 1].fuzzy_predicates)
            else:
                n_fuzzy_predicates = len(
                    self.summarizers[i - self.n_qualifiers - 1].fuzzy_predicates
                )
            for j in range(n_fuzzy_predicates):
                if j == node_code[i]:
                    continue
                sibling_code = list(node_code)
                sibling_code[i] = j
                sibling_id = self._node_id_from_code(
                    data_category,
                    tuple(sibling_code),
                    model_names,
                )
                if sibling_id in node_data:
                    siblings.add(sibling_id)
        return siblings

    def _simplify_ls_fast(self, threshold, output_dir, model_names, save_stage_graphs_gexf=False):
        print("Simplifying linguistic summaries...")

        simplified_statements = {}
        for data_category in self.data_categories:
            if (
                (len(model_names) == 1) and (data_category.model_name in model_names)
            ) or (
                (len(model_names) == 2) and (data_category.model_name == model_names[0])
            ):
                self._scale_aggregated_for_simplification(data_category.index)

                print(f"Stage 1 - Threshold based on Truth Value for {data_category.category_name}...")
                t0 = time.time()
                node_order = []
                node_data = {}
                for q, r, p in self._iter_valid_statement_triples(data_category):
                    node_id = self.get_linguistic_statement(
                        data_category,
                        [q],
                        [r],
                        [p],
                        False,
                        model_names,
                    )
                    if node_id not in node_data:
                        node_order.append(node_id)
                    node_data[node_id] = {
                        'data_type': data_category.index,
                        'q': q,
                        'r': r,
                        'p': p,
                        'v_s': self.aggregated_values[data_category.index][q, r, p],
                        't_s': self.truth_vals[data_category.index][q, r, p],
                        'f_s': self.focus_vals[data_category.index][q, r, p],
                        'o_s': self.operational_relevancy_vals[data_category.index][q, r, p],
                        'c_s': self.simplicity_vals[data_category.index][q, r, p],
                    }

                T = {
                    node_id: node_data[node_id]['v_s'] >= threshold
                    for node_id in node_order
                }
                true_nodes = {node_id for node_id, is_true in T.items() if is_true}
                total_cnt = len(node_order) + 1
                true_cnt = len(true_nodes)
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                print(f"Stage 1 - Determine what needs to be reported for {data_category.category_name}...")
                t0 = time.time()
                R = {}
                if self.params["REMOVE_CHILDREN_FROM_TRUE_PARENTS"]:
                    for node_id in node_order:
                        if not T[node_id]:
                            R[node_id] = False
                            continue
                        data = node_data[node_id]
                        node_code = tuple(self.get_node_code(data['q'], data['r'], data['p']))
                        R[node_id] = not self._has_true_ancestor(
                            data_category,
                            node_code,
                            true_nodes - {node_id},
                            model_names,
                        )
                else:
                    R = T.copy()

                report_cnt = sum(1 for value in R.values() if value)
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                print("\n\n\nAfter Stage 2:")
                cur_S = [node_id for node_id in node_order if R[node_id]]
                for s in cur_S:
                    print(s)
                print("\n\n\n")

                print(f"Stage 3 - Grouping statements to report for {data_category.category_name}...")
                t0 = time.time()

                processed = {node_id: False for node_id in node_order}
                order_index = {node_id: i for i, node_id in enumerate(node_order)}

                def select_node_stage_2():
                    result_ids = [
                        node_id for node_id in node_order
                        if T[node_id] and R[node_id] and (not processed[node_id])
                    ]
                    return random.choice(result_ids) if result_ids else False

                S = []
                cur_node = select_node_stage_2()
                while cur_node:
                    cur_data = node_data[cur_node]
                    cur_node_code = tuple(self.get_node_code(
                        cur_data['q'],
                        cur_data['r'],
                        cur_data['p'],
                    ))
                    siblings = self._close_sibling_node_ids(
                        data_category,
                        cur_node_code,
                        node_data,
                        model_names,
                    )
                    siblings_to_report = [
                        sibling for sibling in siblings
                        if R.get(sibling, False)
                    ]

                    if len(siblings_to_report) == 0:
                        S.append(cur_node)
                        processed[cur_node] = True
                    else:
                        siblings.add(cur_node)
                        siblings_to_report_not_yet_processed = [
                            sibling for sibling in siblings
                            if R.get(sibling, False) and (not processed[sibling])
                        ]
                        siblings_to_report_not_yet_processed = sorted(
                            siblings_to_report_not_yet_processed,
                            key=lambda node_id: order_index[node_id],
                        )
                        siblings_not_to_report_or_already_processed = [
                            sibling for sibling in siblings
                            if not (sibling in siblings_to_report)
                        ]

                        if len(siblings_to_report_not_yet_processed) > 0:
                            use_negative_statement = (
                                self.params["USE_NEGATIVE_STATEMENTS"]
                                and (
                                    len(siblings_to_report_not_yet_processed)
                                    > len(siblings_not_to_report_or_already_processed)
                                )
                                and (len(siblings_not_to_report_or_already_processed) > 0)
                            )

                            qs = []
                            rs = []
                            ps = []
                            processed[cur_node] = True
                            for sibling in siblings_to_report_not_yet_processed:
                                processed[sibling] = True
                                qs.append(node_data[sibling]['q'])
                                rs.append(node_data[sibling]['r'])
                                ps.append(node_data[sibling]['p'])

                            list_A, list_B = find_partitions(ps, rs)

                            qsA = []
                            rsA = []
                            psA = []
                            qsB = []
                            rsB = []
                            psB = []
                            for indx in list_A:
                                qsA.append(qs[indx])
                                rsA.append(rs[indx])
                                psA.append(ps[indx])
                            for indx in list_B:
                                qsB.append(qs[indx])
                                rsB.append(rs[indx])
                                psB.append(ps[indx])

                            if len(qsA) > 0:
                                S.append(self.get_linguistic_statement(
                                    data_category,
                                    qsA,
                                    rsA,
                                    psA,
                                    use_negative_statement,
                                    model_names,
                                ))
                            if len(qsB) > 0:
                                S.append(self.get_linguistic_statement(
                                    data_category,
                                    qsB,
                                    rsB,
                                    psB,
                                    use_negative_statement,
                                    model_names,
                                ))

                    cur_node = select_node_stage_2()

                stage_3_cnt = len(S)
                S = sorted(S)
                t1 = time.time()

                print("\n\n\nAfter Stage 3:")
                for s in S:
                    print(s)
                print("\n\n\n")

                self.results["simplified_stage_3_summary"] = S.copy()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                print(f"Stage 4 - Combining similar statements for {data_category.category_name}...")
                t0 = time.time()
                S = combine_statements(S, self.quantifiers.fuzzy_predicates)
                simplified_statements[data_category.category_name] = S

                print("\n\n\nAfter Stage 4:")
                for s in S:
                    print(s)
                print("\n\n\n")

                self.results["simplified_stage_4_summary"] = S.copy()

                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                print(f"\n\nPrinting simplified LS for {data_category.category_name}")
                print(f"Started with {total_cnt} total statements.")
                print(f"Identified {true_cnt} statements with truth above the threshold.")
                print(f"Of those {report_cnt} were determined to need to be reported.")
                print(f"Resulting in {len(simplified_statements[data_category.category_name])} final simplified statements:\n")

                self.all_statements_cnt.append(total_cnt)
                self.all_statements_after_stage_1_cnt.append(true_cnt)
                self.all_statements_after_stage_2_cnt.append(report_cnt)
                self.all_statements_after_stage_3_cnt.append(stage_3_cnt)
                self.all_statements_after_stage_4_cnt.append(len(simplified_statements[data_category.category_name]))
                self.all_statements_data_types.append(data_category.category_name)

        print("Saving linguitic summary to txt file...")
        t0 = time.time()
        output_filename = output_dir + '/simplified_ls.txt'
        file = open(output_filename, "w")
        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                file.write(f"{data_category.model_name} {data_category.category_name}:\n")
                for i in range(len(simplified_statements[data_category.category_name])):
                    file.write(simplified_statements[data_category.category_name][i] + '\n')
                file.write("\n")
        file.close()
        print("Saved simplified linguistic txt summary: " + output_filename)
        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def simplify_ls(self, threshold, output_dir, model_names, save_stage_graphs_gexf=False):
        if not save_stage_graphs_gexf:
            return self._simplify_ls_fast(
                threshold,
                output_dir,
                model_names,
                save_stage_graphs_gexf=save_stage_graphs_gexf,
            )

        print("Simplifying linguistic summaries...")

        simplified_statements = {}
        for data_category in self.data_categories:
            if (
                (len(model_names) == 1) and (data_category.model_name in model_names)
            ) or (
                (len(model_names) == 2) and (data_category.model_name == model_names[0])
            ):
                print(f"Converting to networkx structure and applying heirarchical layout...")
                t0 = time.time()
                G = self.convert_to_networkx(data_category.index)
                root_node = find_root_node(G)
                G = apply_hierarchical_layout(G, root_node)
                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                # ========= Stage 1: "Threshold" ===================
                print(f"Stage 1 - Threshold based on Truth Value for {data_category.category_name}...")
                t0 = time.time()

                # Keep only nodes with truth value above user-defined threshold.
                total_cnt = 0
                true_cnt = 0
                for node_id in G:
                    total_cnt += 1
                    if G.nodes[node_id]['data']['v_s'] >= threshold:
                        G.nodes[node_id]['data']['T'] = True
                        true_cnt += 1
                    else:
                        G.nodes[node_id]['data']['T'] = False

                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                # Save results from this stage.
                G_copy = G.copy()
                data_type = data_category.category_name.replace(' ', '_')
                out_filename = output_dir + f'/simplified_after_stage_0_{data_type}.gexf'
                off_color = {'r': 255, 'g': 128, 'b': 128}
                on_color = {'r': 128, 'g': 255, 'b': 128}
                for node_id in G_copy:
                    if G_copy.nodes[node_id]['data']['T']:
                        G_copy.nodes[node_id]['viz']['color'] = on_color
                    else:
                        G_copy.nodes[node_id]['viz']['color'] = off_color
                nx.write_gexf(G_copy, out_filename)
                print(f"Wrote to {out_filename}")

                print(f"Stage 1 - Determine what needs to be reported for {data_category.category_name}...")
                t0 = time.time()

                # =========== Stage 2: "What to report" =================
                for node_id in G:
                    if node_id != root_node:
                        if (G.nodes[node_id]['data']['T'] == True):
                            # Remove all children if parent is true
                            if self.params["REMOVE_CHILDREN_FROM_TRUE_PARENTS"]:
                                descendants = nx.descendants(G, node_id)
                                for descendant in descendants:
                                    G.nodes[descendant]['data']['R'] = False
                            else:
                                # If all children are true, remove them and only report parent.
                                children = list(G.successors(node_id))
                                if children and all(G.nodes[child]['data']['T'] for child in children):
                                    for child in children:
                                        G.nodes[child]['data']['R'] = False

                # Report anything left that meets threshold and not already decided to report.
                for node_id in G:
                    if (G.nodes[node_id]['data']['T'] == True) and (G.nodes[node_id]['data']['R'] == None):
                        G.nodes[node_id]['data']['R'] = True
                    else:
                        G.nodes[node_id]['data']['R'] = False

                report_cnt = 0
                for node_id in G:
                    if G.nodes[node_id]['data']['R']:
                        report_cnt += 1

                # Save results for this stage.
                G_copy = G.copy()
                data_type = data_category.category_name.replace(' ', '_')
                out_filename = output_dir + f'/simplified_after_stage_1_{data_type}.gexf'
                for node_id in G_copy:
                    if G_copy.nodes[node_id]['data']['R']:
                        G_copy.nodes[node_id]['viz']['color'] = on_color
                    else:
                        G_copy.nodes[node_id]['viz']['color'] = off_color
                nx.write_gexf(G_copy, out_filename)
                print(f"Wrote to {out_filename}")

                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                print("\n\n\nAfter Stage 2:")
                cur_S = []
                for node_id in G:
                    if G_copy.nodes[node_id]['data']['R']:
                        cur_S.append(self.get_linguistic_statement(data_category, [G.nodes[node_id]['data']['q']], [G.nodes[node_id]['data']['r']], [G.nodes[node_id]['data']['p']], False, model_names))
                for s in cur_S:
                    print(s)
                print("\n\n\n")

                # ================ Stage 3: "How to report?" =====================
                print(f"Stage 3 - Grouping statements to report for {data_category.category_name}...")
                t0 = time.time()

                def select_node_stage_2(G):
                    result_ids = [
                        node_id for node_id in G
                        if (G.nodes[node_id]['data']['T'] == True) and
                        (G.nodes[node_id]['data']['R'] == True) and
                        (G.nodes[node_id]['data']['P'] == False)
                    ]

                    return random.choice(result_ids) if result_ids else False

                S = []
                S_debug = []
                cur_node = select_node_stage_2(G)
                while cur_node:
                    # Find the siblings of the current node, not including the current node.
                    parents = set(G.predecessors(cur_node))
                    siblings = set()
                    for parent in parents:
                        siblings.update(G.successors(parent))
                    siblings.discard(cur_node)

                    # Find all the "close siblings" where only one summarizer or qualifier component differs from the current node.
                    close_siblings = set()
                    cur_node_code = self.get_node_code(G.nodes[cur_node]['data']['q'], G.nodes[cur_node]['data']['r'], G.nodes[cur_node]['data']['p'])
                    for sibling in siblings:
                        sibling_node_code = self.get_node_code(G.nodes[sibling]['data']['q'], G.nodes[sibling]['data']['r'], G.nodes[sibling]['data']['p'])

                        differences = 0
                        for i in range(len(cur_node_code)):
                            if cur_node_code[i] != sibling_node_code[i]:
                                differences += 1
                        if differences == 1:
                            close_siblings.add(sibling)

                    siblings = close_siblings

                    # Get a list of all the siblings that should be reported (determined from previous stages)
                    siblings_to_report = [sibling for sibling in siblings if G.nodes[sibling]['data']['R']]

                    if len(siblings_to_report) == 0:
                        # If there are no siblings that need to be reported then only report the current node.
                        S.append(cur_node)
                        G.nodes[cur_node]['data']['P'] = True

                        S_debug.append([])
                        S_debug[-1].append(cur_node)
                    else:
                        # Add the current node back to the list of siblings.
                        siblings.add(cur_node)

                        # Get a list of the siblings that should be reported and have not yet been processed.
                        siblings_to_report_not_yet_processed = [
                            sibling for sibling in siblings
                            if G.nodes[sibling]['data']['R'] and (not G.nodes[sibling]['data']['P'])
                        ]

                        # Get a list of the siblings that
                        siblings_not_to_report_or_already_processed = [
                            sibling for sibling in siblings
                            if not (sibling in siblings_to_report)
                        ]

                        if len(siblings_to_report_not_yet_processed) > 0:
                            # Use a negation of the statement if it is simpler (and allowed based on parameters)
                            if self.params["USE_NEGATIVE_STATEMENTS"] and (len(siblings_to_report_not_yet_processed) > len(siblings_not_to_report_or_already_processed)) and (len(siblings_not_to_report_or_already_processed) > 0):
                                use_negative_statement = True
                            else:
                                use_negative_statement = False

                            # Group the summarizer, qualifier, and quantifier indices of each sibling to report. Also mark the current node and siblings that will be reported to processed.
                            qs = []
                            rs = []
                            ps = []
                            G.nodes[cur_node]['data']['P'] = True
                            for sibling in siblings_to_report_not_yet_processed:
                                G.nodes[sibling]['data']['P'] = True

                                qs.append(G.nodes[sibling]['data']['q'])
                                rs.append(G.nodes[sibling]['data']['r'])
                                ps.append(G.nodes[sibling]['data']['p'])

                            # Determine the linguistic statment associated with the group.
                            # For simplicity, split into two statements, one with all summarizers euqal and one with all qualifiers equal
                            # summarizers = []
                            # qualifiers = []
                            # for indx in range(len(ps)):

                            list_A, list_B = find_partitions(ps, rs)

                            qsA = []
                            rsA = []
                            psA = []
                            qsB = []
                            rsB = []
                            psB = []
                            for indx in list_A:
                                qsA.append(qs[indx])
                                rsA.append(rs[indx])
                                psA.append(ps[indx])
                            for indx in list_B:
                                qsB.append(qs[indx])
                                rsB.append(rs[indx])
                                psB.append(ps[indx])

                            # for indx in range(len(ps)):
                                # print(f"q = {qs[indx]}, p = {ps[indx]}, r = {rs[indx]} | " +  str(self.get_node_code(qs[indx], rs[indx], ps[indx])) + " | " + self.get_linguistic_statement(data_category, [qs[indx]], [rs[indx]], [ps[indx]], use_negative_statement, model_names))

                            # for indx in range(len(psA)):
                                # print(str(self.get_node_code(qsA[indx], rsA[indx], psA[indx])) + self.get_linguistic_statement(data_category, [qsA[indx]], [rsA[indx]], [psA[indx]], use_negative_statement, model_names))

                            if len(qsA) > 0:
                                S.append(self.get_linguistic_statement(data_category, qsA, rsA, psA, use_negative_statement, model_names))
                                # print(f"Combined Statement: {S[-1]}")

                            # for indx in range(len(psB)):
                                # print(str(self.get_node_code(qsB[indx], rsB[indx], psB[indx])) + self.get_linguistic_statement(data_category, [qsB[indx]], [rsB[indx]], [psB[indx]], use_negative_statement, model_names))

                            if len(qsB) > 0:
                                S.append(self.get_linguistic_statement(data_category, qsB, rsB, psB, use_negative_statement, model_names))
                                # print(f"Combined Statement: {S[-1]}")

                    cur_node = select_node_stage_2(G)

                stage_3_cnt = len(S)
                S = sorted(S)

                print("\n\n\nAfter Stage 3:")
                for s in S:
                    print(s)
                print("\n\n\n")

                self.results["simplified_stage_3_summary"] = S.copy()

                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                # ================ Stage 4: "Combine similar statements" =====================
                print(f"Stage 4 - Combining similar statements for {data_category.category_name}...")

                S =  combine_statements(S, self.quantifiers.fuzzy_predicates)
                simplified_statements[data_category.category_name] = S

                print("\n\n\nAfter Stage 4:")
                for s in S:
                    print(s)
                print("\n\n\n")

                self.results["simplified_stage_4_summary"] = S.copy()

                t1 = time.time()
                print("Done. Took {}s.".format(round(t1 - t0, 3)))

                print(f"\n\nPrinting simplified LS for {data_category.category_name}")
                print(f"Started with {total_cnt} total statements.")
                print(f"Identified {true_cnt} statements with truth above the threshold.")
                print(f"Of those {report_cnt} were determined to need to be reported.")
                print(f"Resulting in {len(simplified_statements[data_category.category_name])} final simplified statements:\n")

                self.all_statements_cnt.append(total_cnt)
                self.all_statements_after_stage_1_cnt.append(true_cnt)
                self.all_statements_after_stage_2_cnt.append(report_cnt)
                self.all_statements_after_stage_3_cnt.append(stage_3_cnt)
                self.all_statements_after_stage_4_cnt.append(len(simplified_statements[data_category.category_name]))
                self.all_statements_data_types.append(data_category.category_name)

        print("Saving linguitic summary to txt file...")
        t0 = time.time()
        output_filename = output_dir + '/simplified_ls.txt'
        file = open(output_filename, "w")
        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                file.write(f"{data_category.model_name} {data_category.category_name}:\n")
                for i in range(len(simplified_statements[data_category.category_name])):
                    file.write(simplified_statements[data_category.category_name][i] + '\n')
                file.write("\n")
        file.close()
        print("Saved simplified linguistic txt summary: " + output_filename)
        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def export_graph(self, output_filename, model_names):
        print("Exporting graph...")
        t0 = time.time()
        for data_category in self.data_categories:
            if (
                len(model_names) == 1 and data_category.model_name in model_names
            ) or (
                len(model_names) == 2 and data_category.model_name == model_names[0]
            ):
                output_filename_full = output_filename + f'_{data_category.category_name.replace(" ", "_")}_{data_category.model_name}.gexf'

                nx_graph = self.convert_to_networkx(data_category.index)

                # Find the root node dynamically
                root_node = find_root_node(nx_graph)

                nx_graph = apply_hierarchical_layout(nx_graph, root_node)

                for node_id in nx_graph:
                    v_s = nx_graph.nodes[node_id]['data']['v_s']
                    nx_graph.nodes[node_id]['label'] = f"{v_s:.3f} | " + nx_graph.nodes[node_id]['label']

                nx.write_gexf(nx_graph, output_filename_full)

                print('Saved graph output to', output_filename_full)
        t1 = time.time()
        print("Done. Took {}s.".format(round(t1 - t0, 3)))

    def print_results(self):
        # Print initial summary:
        print("\nPrinting initial summary:")
        for i in range(len(self.results["initial_linguistic_summary"]["linguistic_statements"])):
            s = self.results["initial_linguistic_summary"]["linguistic_statements"][i]
            t = self.results["initial_linguistic_summary"]["truth_vals"][i]
            # if t > 0:
            print(f"{t:0.3} | {s}")

        # Print intermediate simplified summary:
        print("\nPrinting intermediate simplified summary:")
        for s in self.results["simplified_stage_3_summary"]:
            print(s)

        # Print final simplified summary:
        print("\nPrinting final simplified summary:")
        for s in self.results["simplified_stage_4_summary"]:
            print(s)

    def print_all_stats(self):
        print_stats(0, self.all_statements_cnt)
        print_stats(1, self.all_statements_after_stage_1_cnt)
        print_stats(2, self.all_statements_after_stage_2_cnt)
        print_stats(3, self.all_statements_after_stage_3_cnt)
        print_stats(4, self.all_statements_after_stage_4_cnt)
