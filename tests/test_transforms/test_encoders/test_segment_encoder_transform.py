import numpy as np
import pandas as pd

from etna.transforms import SegmentEncoderTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


def test_segment_encoder_transform(simple_ts):
    transform = SegmentEncoderTransform()
    transformed_df = transform.fit_transform(simple_ts).to_pandas()
    assert (
        len(transformed_df.loc[:, pd.IndexSlice[:, "segment_code"]].columns) == 2
    ), "Number of columns not the same as segments"
    assert len(simple_ts.to_pandas()) == len(transformed_df), "Row missing"
    codes = set()
    for segment in simple_ts.columns.get_level_values("segment").unique():
        column = transformed_df.loc[:, pd.IndexSlice[segment, "segment_code"]]
        assert column.dtype == "category", "Column type is not category"
        assert np.all(column == column.iloc[0]), "Values are not the same for the whole column"
        codes.add(column.iloc[0])
    assert codes == {0, 1}, "Codes are not 0 and 1"


def test_save_load(example_tsds):
    transform = SegmentEncoderTransform()
    assert_transformation_equals_loaded_original(transform=transform, ts=example_tsds)
