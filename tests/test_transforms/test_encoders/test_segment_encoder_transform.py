import numpy as np
import pandas as pd

from etna.transforms import SegmentEncoderTransform


def test_segment_encoder_transform(simple_ts):
    transform = SegmentEncoderTransform()
    transformed_ts = transform.fit_transform(simple_ts)
    assert (
        len(transformed_ts.loc[:, pd.IndexSlice[:, "segment_code"]].columns) == 2
    ), "Number of columns not the same as segments"
    assert len(simple_ts.to_pandas()) == len(transformed_ts.to_pandas()), "Row missing"
    codes = set()
    for segment in simple_ts.columns.get_level_values("segment").unique():
        column = transformed_ts.to_pandas().loc[:, pd.IndexSlice[segment, "segment_code"]]
        assert column.dtype == "category", "Column type is not category"
        assert np.all(column == column.iloc[0]), "Values are not the same for the whole column"
        codes.add(column.iloc[0])
    assert codes == {0, 1}, "Codes are not 0 and 1"
