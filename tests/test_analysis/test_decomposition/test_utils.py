import pytest

from etna.analysis.decomposition.utils import _get_labels_names
from etna.transforms import LinearTrendTransform
from etna.transforms import TheilSenTrendTransform


@pytest.mark.parametrize(
    "poly_degree, expect_values, trend_class",
    (
        [1, True, LinearTrendTransform],
        [2, False, LinearTrendTransform],
        [1, True, TheilSenTrendTransform],
        [2, False, TheilSenTrendTransform],
    ),
)
def test_get_labels_names_linear_coeffs(example_tsdf, poly_degree, expect_values, trend_class):
    ln_tr = trend_class(in_column="target", poly_degree=poly_degree)
    ln_tr.fit_transform(example_tsdf)
    segments = example_tsdf.segments
    _, linear_coeffs = _get_labels_names([ln_tr], segments)
    if expect_values:
        assert list(linear_coeffs.values()) != ["", ""]
    else:
        assert list(linear_coeffs.values()) == ["", ""]
