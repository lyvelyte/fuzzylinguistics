import numpy as np
import pandas as pd
import pytest

from fuzzylinguistics import (
    DatasetConfig,
    FuzzyLinguisticSummaries,
    setup_fls,
    setup_fls_from_data,
)
from fuzzylinguistics import fuzzy_linguistic_summaries as fls_module
from fuzzylinguistics.fuzzy_linguistic_summaries import (
    evaluate_trapezoidal_fuzzy_membership,
)


def test_trapezoidal_membership_edges_and_nan_quantifier():
    x = np.array([0.0, 0.5, 1.0, 2.0])
    values = evaluate_trapezoidal_fuzzy_membership([0.0, 0.5, 1.0, 2.0], x)
    np.testing.assert_allclose(values, [0.0, 1.0, 1.0, 0.0])

    nan_value = evaluate_trapezoidal_fuzzy_membership(
        [np.nan, np.nan, np.nan, np.nan],
        np.array(float("nan")),
    )
    np.testing.assert_allclose(nan_value, 1.0)


def test_backend_selection_falls_back_and_errors(monkeypatch):
    monkeypatch.setattr(
        fls_module,
        "_probe_cupy_available",
        lambda: (False, "test unavailable"),
    )
    xp, name = fls_module._resolve_array_backend("auto")
    assert xp is np
    assert name == "numpy"

    with pytest.raises(RuntimeError, match="CuPy"):
        fls_module._resolve_array_backend("cupy")


def test_simple_config_batch_equivalence(tmp_path):
    fls_normal = setup_fls("examples/configs/config_simple_single.json")
    result_normal = fls_normal.generate_fls_one_model(
        results_dir=str(tmp_path / "normal"),
        compute_backend="numpy",
        save_initial_fls_txt=False,
        simplify_fls=True,
    )

    fls_batched = setup_fls("examples/configs/config_simple_single.json")
    result_batched = fls_batched.generate_fls_one_model(
        results_dir=str(tmp_path / "batched"),
        compute_backend="numpy",
        max_working_memory_mb=1,
        statement_batch_size=1,
        save_initial_fls_txt=False,
        simplify_fls=True,
    )

    assert (
        result_batched["simplified_stage_4_summary"]
        == result_normal["simplified_stage_4_summary"]
    )
    np.testing.assert_allclose(
        fls_batched.sorted_truth_vals,
        fls_normal.sorted_truth_vals,
    )
    np.testing.assert_allclose(
        fls_batched.sorted_aggregated_values,
        fls_normal.sorted_aggregated_values,
    )


def test_graph_export_disabled_by_default_and_opt_in(tmp_path):
    fls_default = setup_fls("examples/configs/config_simple_single.json")
    fls_default.generate_fls_one_model(
        results_dir=str(tmp_path / "default"),
        compute_backend="numpy",
        save_initial_fls_txt=False,
        simplify_fls=True,
    )
    assert not list((tmp_path / "default").glob("*.gexf"))

    fls_graph = setup_fls("examples/configs/config_simple_single.json")
    fls_graph.generate_fls_one_model(
        results_dir=str(tmp_path / "graph"),
        compute_backend="numpy",
        save_initial_fls_txt=False,
        simplify_fls=True,
        save_simplification_graph_gexf=True,
    )
    assert list((tmp_path / "graph").glob("*.gexf"))


def test_no_qualifier_generation(tmp_path):
    fls = FuzzyLinguisticSummaries()
    input_data = np.array([[0.0], [0.0], [1.0], [1.0]])
    output_data = np.zeros((4, 0))
    fls.add_data_category(
        "Items",
        False,
        input_data,
        ["score"],
        [None],
        output_data,
        [],
        [],
        "Model",
    )
    fls.add_summarizer(
        "score",
        "score",
        ["low", "high"],
        np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=float),
        [1.0, 1.0],
    )
    fls.add_quantifiers(
        ["None", "Some", "All"],
        np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.5, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        [1.0, 1.0, 1.0],
    )

    result = fls.generate_fls_one_model(
        results_dir=str(tmp_path),
        compute_backend="numpy",
        save_initial_fls_txt=False,
        simplify_fls=False,
    )
    assert len(result["initial_linguistic_summary"]["linguistic_statements"]) == 9


def test_setup_fls_from_data_accepts_pandas_and_lists():
    df = pd.read_csv("examples/data.csv")
    input_df = df.iloc[:, :2]
    output_data = df.iloc[:, 2:].values.tolist()

    dataset = DatasetConfig(
        category_name="Cars",
        model_name="Car Example",
        uses_qualifier=True,
        input_dimension_labels=list(input_df.columns),
        input_dimension_units=[None] * input_df.shape[1],
        input_data=input_df,
        output_dimension_labels=list(df.columns[2:]),
        output_dimension_units=[None] * (df.shape[1] - 2),
        output_data=output_data,
    )

    fls = setup_fls_from_data(
        [dataset],
        "examples/configs/membership_functions.json",
    )

    assert isinstance(fls.data_categories[0].input_data, np.ndarray)
    assert isinstance(fls.data_categories[0].output_data, np.ndarray)


def test_many_column_memory_safe_streaming(tmp_path):
    n_records = 6
    n_columns = 9
    data = np.tile(np.array([[0.0, 1.0]]).T, (3, n_columns))

    fls = FuzzyLinguisticSummaries()
    fls.add_data_category(
        "Wide",
        False,
        data,
        [f"x{i}" for i in range(n_columns)],
        [None] * n_columns,
        np.zeros((n_records, 0)),
        [],
        [],
        "Model",
    )
    for i in range(n_columns):
        fls.add_summarizer(
            f"x{i}",
            f"x{i}",
            ["low", "high"],
            np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=float),
            [1.0, 1.0],
        )
    fls.add_quantifiers(
        ["None", "All"],
        np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        [1.0, 1.0],
    )

    result = fls.generate_fls_one_model(
        results_dir=str(tmp_path),
        compute_backend="numpy",
        max_working_memory_mb=1,
        statement_batch_size=11,
        initial_summary_mode="auto",
        max_initial_summary_items=10,
        save_initial_fls_txt=False,
        simplify_fls=False,
    )

    assert fls.initial_summary_streamed
    assert fls.n_summarizer_statements == 3 ** n_columns
    assert result["initial_linguistic_summary"]["linguistic_statements"].size == 0


def test_cupy_backend_matches_numpy_when_available(tmp_path):
    available, reason = fls_module._probe_cupy_available()
    if not available:
        pytest.skip(reason)

    fls_numpy = setup_fls("examples/configs/config_simple_single.json")
    fls_numpy.generate_fls_one_model(
        results_dir=str(tmp_path / "numpy"),
        compute_backend="numpy",
        save_initial_fls_txt=False,
        simplify_fls=False,
    )

    fls_cupy = setup_fls("examples/configs/config_simple_single.json")
    fls_cupy.generate_fls_one_model(
        results_dir=str(tmp_path / "cupy"),
        compute_backend="cupy",
        save_initial_fls_txt=False,
        simplify_fls=False,
    )

    np.testing.assert_allclose(
        fls_cupy.sorted_truth_vals,
        fls_numpy.sorted_truth_vals,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fls_cupy.sorted_aggregated_values,
        fls_numpy.sorted_aggregated_values,
        atol=1e-12,
    )
