import warnings

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from shapely.geometry import Point
from spatialdata.models import PointsModel, ShapesModel, TableModel

import segtraq as st


def _make_sdata(cell_ids):
    n = len(cell_ids)
    points = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "y": np.arange(n, dtype=float),
            "feature_name": pd.Categorical(["A"] * n),
            "cell_id": pd.Series(cell_ids, dtype=object),
        }
    )
    cells = ["c1", "c2"]
    shapes = gpd.GeoDataFrame(
        {"cell_id": cells},
        geometry=[Point(0, 0).buffer(1.0), Point(5, 5).buffer(1.0)],
    )
    table = ad.AnnData(
        X=np.zeros((len(cells), 1)),
        obs=pd.DataFrame({"cell_id": cells, "region": pd.Categorical(["cell_boundaries"] * len(cells))}, index=cells),
        var=pd.DataFrame(index=["A"]),
    )
    return sd.SpatialData(
        points={"transcripts": PointsModel.parse(points)},
        shapes={"cell_boundaries": ShapesModel.parse(shapes)},
        tables={
            "table": TableModel.parse(table, region="cell_boundaries", region_key="region", instance_key="cell_id")
        },
    )


def _validate(sdata, points_background_id):
    return st.validate_spatialdata(
        sdata,
        images_key=None,
        tables_centroid_x_key=None,
        tables_centroid_y_key=None,
        tables_area_key=None,
        points_z_key=None,
        nucleus_shapes_key=None,
        points_background_id=points_background_id,
    )


# most transcripts are unassigned, encoded as missing cell IDs
MOSTLY_MISSING = ["c1", "c2", "c2", None, None, None]


def test_validate_missing_background_not_found_suggests_none():
    sdata = _make_sdata(MOSTLY_MISSING)
    with pytest.raises(AssertionError) as excinfo:
        _validate(sdata, points_background_id=-1)
    msg = str(excinfo.value)
    assert "missing value" in msg
    assert "points_background_id=None" in msg
    # the most common non-missing cell ID must not be reported as the most common one
    assert "'c2'" not in msg


def test_validate_missing_background_not_most_common_suggests_none():
    sdata = _make_sdata(MOSTLY_MISSING)
    with pytest.warns(UserWarning, match="points_background_id=None"):
        _validate(sdata, points_background_id="c1")


@pytest.mark.parametrize("background_id", [None, np.nan, pd.NA])
def test_validate_missing_background_ids_are_equivalent(background_id):
    sdata = _make_sdata(MOSTLY_MISSING)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*points_background_id.*")
        _validate(sdata, points_background_id=background_id)


def test_validate_explicit_background_most_common_no_warning():
    sdata = _make_sdata(["c1", "c2", "UNASSIGNED", "UNASSIGNED", "UNASSIGNED"])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*points_background_id.*")
        _validate(sdata, points_background_id="UNASSIGNED")


def test_validate_explicit_background_not_most_common_reports_most_common():
    sdata = _make_sdata(["c1", "c2", "c2", "c2", "UNASSIGNED"])
    with pytest.warns(UserWarning, match="most common cell ID among points is 'c2'"):
        _validate(sdata, points_background_id="UNASSIGNED")
