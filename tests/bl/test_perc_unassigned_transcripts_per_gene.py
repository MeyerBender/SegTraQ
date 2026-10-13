import anndata as ad
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from spatialdata.models import PointsModel, TableModel

import segtraq as st


def test_perc_unassigned_transcripts_per_gene(sdata_new):
    perc_unassigned = st.bl.perc_unassigned_transcripts_per_gene(sdata_new)
    arr = perc_unassigned["perc_unassigned"].values
    assert np.all((arr >= 0) & (arr <= 100)), "Percentage of unassigned transcripts should be between 0 and 100"
    assert "perc_unassigned" in sdata_new.tables["table"].var.columns, "'perc_unassigned' should be present in var"


def _make_sdata(cell_ids, genes):
    points = pd.DataFrame(
        {
            "x": np.arange(len(genes), dtype=float),
            "y": np.arange(len(genes), dtype=float),
            "feature_name": pd.Categorical(genes),
            "cell_id": cell_ids,
        }
    )
    gene_names = sorted(set(genes))
    table = ad.AnnData(
        X=np.zeros((2, len(gene_names))),
        obs=pd.DataFrame({"cell_id": ["c1", "c2"]}, index=["0", "1"]),
        var=pd.DataFrame(index=gene_names),
    )
    return sd.SpatialData(
        points={"transcripts": PointsModel.parse(points)},
        tables={"table": TableModel.parse(table)},
    )


@pytest.mark.parametrize("background_id", [None, np.nan])
def test_perc_unassigned_transcripts_per_gene_missing_background(background_id):
    # unassigned transcripts are encoded as missing cell ids
    genes = ["A", "A", "A", "A", "B", "B", "C"]
    cell_ids = pd.Series(["c1", None, None, None, "c2", "c1", None], dtype=object)
    sdata = _make_sdata(cell_ids, genes)

    result = st.bl.perc_unassigned_transcripts_per_gene(sdata, points_background_id=background_id)

    assert result.loc["A", "total"] == 4
    assert result.loc["A", "unassigned"] == 3
    assert result.loc["A", "perc_unassigned"] == pytest.approx(75.0)
    assert result.loc["B", "perc_unassigned"] == pytest.approx(0.0)
    assert result.loc["C", "perc_unassigned"] == pytest.approx(100.0)
    arr = result["perc_unassigned"].values
    assert np.all((arr >= 0) & (arr <= 100)), "Percentage of unassigned transcripts should be between 0 and 100"


def test_perc_unassigned_transcripts_per_gene_string_background():
    genes = ["A", "A", "A", "B", "B"]
    cell_ids = pd.Series(["c1", "UNASSIGNED", "UNASSIGNED", "c2", "UNASSIGNED"], dtype=object)
    sdata = _make_sdata(cell_ids, genes)

    result = st.bl.perc_unassigned_transcripts_per_gene(sdata, points_background_id="UNASSIGNED")

    assert result.loc["A", "perc_unassigned"] == pytest.approx(200 / 3)
    assert result.loc["B", "perc_unassigned"] == pytest.approx(50.0)


def test_perc_unassigned_transcripts_per_gene_invalid_key(sdata_new):
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts_per_gene(sdata_new, points_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts_per_gene(sdata_new, points_cell_id_key="invalid_key")
    with pytest.raises(KeyError):
        st.bl.perc_unassigned_transcripts_per_gene(sdata_new, points_gene_key="invalid_key")
