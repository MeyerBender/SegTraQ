import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geoseries_equal
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from spatialdata import SpatialData

import segtraq as st

st.settings.n_jobs = -1

# tolerances for comparing floating point outputs against the saved reference
RTOL = 1e-5
ATOL = 1e-8

# `.uns` DataFrames whose row order is not deterministic, mapped to the columns that identify a row.
# the gene pairs of the mutually exclusive co-expression rate are collected in a set, so their order
# depends on Python's (randomized) string hashing
UNORDERED_UNS_FRAMES = {"mutually_exclusive_coexpression_rate": ["gene1", "gene2"]}

# the clustering-stability metrics run PCA -> kNN graph -> Leiden. PCA picks up rounding differences
# between scipy versions and CPUs, which can move a few borderline cells into another Leiden cluster.
# exact cluster assignments are therefore not compared, only that the clusterings agree and that the
# derived stability scores are reproducible within a tolerance.
LEIDEN_PREFIX = "leiden_"
LEIDEN_MIN_ARI = 0.95
CLUSTERING_SCORES = ("cluster_connectedness", "silhouette_score", "mean_purity", "mean_ari")
CLUSTERING_SCORE_ATOL = 0.02
PCA_KEY = "X_pca_segtraq"
PCA_ATOL = 1e-3
PCA_RTOL = 1e-3
KNN_GRAPH_KEYS = ("neighbors_segtraq_connectivities", "neighbors_segtraq_distances")
KNN_MIN_EDGE_OVERLAP = 0.95


def _load_generator():
    # tests are imported with `--import-mode=importlib`, so the script cannot be imported by name
    path = Path(__file__).parent / "generate_run_all_reference.py"
    spec = importlib.util.spec_from_file_location("generate_run_all_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_generator = _load_generator()
REFERENCE_PATH = _generator.REFERENCE_PATH
run_all_on_proseg = _generator.run_all_on_proseg


def _assert_array_equal(actual, expected, where):
    if sparse.issparse(actual) or sparse.issparse(expected):
        assert sparse.issparse(actual) and sparse.issparse(expected), f"{where}: sparse/dense mismatch"
        actual, expected = actual.toarray(), expected.toarray()
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"
    try:
        if np.issubdtype(actual.dtype, np.number) and np.issubdtype(expected.dtype, np.number):
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, expected)
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from e


def _assert_frame_equal(actual, expected, where):
    assert list(actual.columns) == list(expected.columns), (
        f"{where}: columns differ.\n"
        f"Only in result: {sorted(set(actual.columns) - set(expected.columns))}\n"
        f"Only in reference: {sorted(set(expected.columns) - set(actual.columns))}"
    )
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=RTOL, atol=ATOL)
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from e


def _assert_equal(actual, expected, where):
    """Recursively compare (nested) containers, as stored in e.g. `.uns`."""
    if isinstance(expected, pd.DataFrame):
        assert isinstance(actual, pd.DataFrame), f"{where}: expected a DataFrame, got {type(actual)}"
        _assert_frame_equal(actual, expected, where)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), f"{where}: expected a mapping, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"{where}: keys differ.\n"
            f"Only in result: {sorted(set(actual.keys()) - set(expected.keys()))}\n"
            f"Only in reference: {sorted(set(expected.keys()) - set(actual.keys()))}"
        )
        for key in expected:
            _assert_equal(actual[key], expected[key], f"{where}[{key!r}]")
    elif isinstance(expected, str | bytes):
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"
    else:
        _assert_array_equal(actual, expected, where)


def _assert_leiden_agrees(actual: pd.Series, expected: pd.Series, where):
    # subset clusterings leave the cells outside the subset unlabeled; the subsets are seeded
    assert (actual.isna() == expected.isna()).all(), f"{where}: different cells are unlabeled"
    labeled = expected.notna()
    ari = adjusted_rand_score(expected[labeled].astype(str), actual[labeled].astype(str))
    assert ari >= LEIDEN_MIN_ARI, f"{where}: adjusted Rand index to the reference is {ari:.3f} < {LEIDEN_MIN_ARI}"


def _assert_pca_close(actual, expected, where):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"
    # the sign of each principal component is arbitrary
    signs = np.where(np.sum(actual * expected, axis=0) < 0, -1.0, 1.0)
    try:
        np.testing.assert_allclose(actual * signs, expected, rtol=PCA_RTOL, atol=PCA_ATOL)
    except AssertionError as e:
        raise AssertionError(f"{where}: {e}") from e


def _assert_knn_graph_close(actual, expected, where):
    assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"
    edges_actual = set(zip(*sparse.coo_matrix(actual).nonzero(), strict=True))
    edges_expected = set(zip(*sparse.coo_matrix(expected).nonzero(), strict=True))
    overlap = len(edges_actual & edges_expected) / max(len(edges_actual | edges_expected), 1)
    assert overlap >= KNN_MIN_EDGE_OVERLAP, (
        f"{where}: only {overlap:.3f} of the kNN graph edges agree with the reference (< {KNN_MIN_EDGE_OVERLAP})"
    )


def _assert_table_equal(actual, expected, where):
    assert actual.shape == expected.shape, f"{where}: shape {actual.shape} != {expected.shape}"

    # clustering outputs are compared with tolerances (see LEIDEN_MIN_ARI), everything else exactly
    assert list(actual.obs.columns) == list(expected.obs.columns), (
        f"{where}.obs: columns differ.\n"
        f"Only in result: {sorted(set(actual.obs.columns) - set(expected.obs.columns))}\n"
        f"Only in reference: {sorted(set(expected.obs.columns) - set(actual.obs.columns))}"
    )
    leiden_cols = [c for c in expected.obs.columns if c.startswith(LEIDEN_PREFIX)]
    for col in leiden_cols:
        _assert_leiden_agrees(actual.obs[col], expected.obs[col], f"{where}.obs[{col!r}]")
    _assert_frame_equal(actual.obs.drop(columns=leiden_cols), expected.obs.drop(columns=leiden_cols), f"{where}.obs")

    _assert_frame_equal(actual.var, expected.var, f"{where}.var")
    _assert_array_equal(actual.X, expected.X, f"{where}.X")

    actual_obsm, expected_obsm = dict(actual.obsm), dict(expected.obsm)
    if PCA_KEY in expected_obsm and PCA_KEY in actual_obsm:
        _assert_pca_close(actual_obsm.pop(PCA_KEY), expected_obsm.pop(PCA_KEY), f"{where}.obsm[{PCA_KEY!r}]")
    _assert_equal(actual_obsm, expected_obsm, f"{where}.obsm")

    actual_obsp, expected_obsp = dict(actual.obsp), dict(expected.obsp)
    for key in KNN_GRAPH_KEYS:
        if key in expected_obsp and key in actual_obsp:
            _assert_knn_graph_close(actual_obsp.pop(key), expected_obsp.pop(key), f"{where}.obsp[{key!r}]")
    _assert_equal(actual_obsp, expected_obsp, f"{where}.obsp")

    for attr in ("layers", "varm", "varp"):
        _assert_equal(dict(getattr(actual, attr)), dict(getattr(expected, attr)), f"{where}.{attr}")

    actual_uns, expected_uns = dict(actual.uns), dict(expected.uns)
    for key, sort_cols in UNORDERED_UNS_FRAMES.items():
        for uns in (actual_uns, expected_uns):
            if isinstance(uns.get(key), pd.DataFrame):
                uns[key] = uns[key].sort_values(sort_cols).reset_index(drop=True)
    for key in CLUSTERING_SCORES:
        if key in expected_uns and key in actual_uns:
            a, e = float(actual_uns.pop(key)), float(expected_uns.pop(key))
            assert abs(a - e) <= CLUSTERING_SCORE_ATOL, (
                f"{where}.uns[{key!r}]: {a:.4f} differs from the reference {e:.4f} by more than {CLUSTERING_SCORE_ATOL}"
            )
    if PCA_KEY in expected_uns and PCA_KEY in actual_uns:
        pca_actual, pca_expected = dict(actual_uns.pop(PCA_KEY)), dict(expected_uns.pop(PCA_KEY))
        for key in ("variance", "variance_ratio"):
            if key in pca_expected and key in pca_actual:
                np.testing.assert_allclose(
                    pca_actual.pop(key),
                    pca_expected.pop(key),
                    rtol=PCA_RTOL,
                    err_msg=f"{where}.uns[{PCA_KEY!r}][{key!r}]",
                )
        _assert_equal(pca_actual, pca_expected, f"{where}.uns[{PCA_KEY!r}]")
    _assert_equal(actual_uns, expected_uns, f"{where}.uns")


def assert_sdata_equal(actual: SpatialData, expected: SpatialData):
    """Assert that the tables, shapes and points of two SpatialData objects match."""
    for element_type in ("images", "labels", "points", "shapes", "tables"):
        assert set(getattr(actual, element_type).keys()) == set(getattr(expected, element_type).keys()), (
            f"{element_type} elements differ: "
            f"{sorted(getattr(actual, element_type).keys())} != {sorted(getattr(expected, element_type).keys())}"
        )

    for key, table in expected.tables.items():
        _assert_table_equal(actual.tables[key], table, f"tables[{key!r}]")

    for key, shapes in expected.shapes.items():
        where = f"shapes[{key!r}]"
        actual_shapes = actual.shapes[key]
        geometry_col = shapes.geometry.name
        _assert_frame_equal(
            pd.DataFrame(actual_shapes.drop(columns=geometry_col)),
            pd.DataFrame(shapes.drop(columns=geometry_col)),
            where,
        )
        try:
            assert_geoseries_equal(actual_shapes.geometry, shapes.geometry, check_less_precise=True)
        except AssertionError as e:
            raise AssertionError(f"{where}.geometry: {e}") from e

    for key, points in expected.points.items():
        _assert_frame_equal(actual.points[key].compute(), points.compute(), f"points[{key!r}]")


# this only tests that run_all works without errors and that it correctly determines
# which modules can and cannot be run, given the arguments it was passed.
def test_run_all_skips_modules_missing_prerequisites(segtraq_obj):
    with pytest.warns(UserWarning, match="run_supervised"):
        result = segtraq_obj.run_all(inplace=False)

    # supervised metrics require either `cell_type_key`+`markers` or a reference dataset;
    # none of these are provided here, so this module should be skipped automatically
    assert "supervised" in result["skipped"]
    assert result["supervised"] is None

    # all other modules do not strictly require a reference and should run successfully
    assert result["baseline"] is not None
    assert "num_cells" in result["baseline"]

    assert result["region_similarity"] is not None
    assert "ious" in result["region_similarity"]

    assert result["volume"] is not None
    assert "similarity_top_bottom" in result["volume"]

    assert result["clustering_stability"] is not None
    assert "cluster_connectedness" in result["clustering_stability"]

    assert result["point_statistics"] is not None
    assert "distance_to_centroid" in result["point_statistics"]

    for name in ("baseline", "region_similarity", "volume", "clustering_stability", "point_statistics"):
        assert name not in result["skipped"]


# runs every module (including the volume metrics) on the 3D ProSeg dataset and compares ALL outputs
# to a previously saved result, generated with `tests/generate_run_all_reference.py`.
# If a metric was changed on purpose, re-generate the reference with that script and upload it.
def test_run_all_matches_reference(tmp_path):
    assert REFERENCE_PATH.exists(), (
        f"Reference output {REFERENCE_PATH} not found. Download the latest test data "
        "or generate it with `python tests/generate_run_all_reference.py`."
    )

    result = run_all_on_proseg()

    # write and re-read the result, so that both objects went through the same zarr (de)serialization
    result.write(tmp_path / "run_all.zarr")
    actual = SpatialData.read(tmp_path / "run_all.zarr")
    expected = SpatialData.read(REFERENCE_PATH)

    assert_sdata_equal(actual, expected)
