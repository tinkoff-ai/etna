import numpy as np


def _get_labels_names(trend_transform, segments):
    """If only unique transform classes are used then show their short names (without parameters). Otherwise show their full repr as label."""
    from etna.transforms.decomposition import LinearTrendTransform
    from etna.transforms.decomposition import TheilSenTrendTransform

    labels = [transform.__repr__() for transform in trend_transform]
    labels_short = [i[: i.find("(")] for i in labels]
    if len(np.unique(labels_short)) == len(labels_short):
        labels = labels_short
    linear_coeffs = dict(zip(segments, ["" for i in range(len(segments))]))
    if (
        len(trend_transform) == 1
        and isinstance(trend_transform[0], (LinearTrendTransform, TheilSenTrendTransform))
        and trend_transform[0].poly_degree == 1
    ):
        for seg in segments:
            linear_coeffs[seg] = (
                ", k=" + f"{trend_transform[0].segment_transforms[seg]._pipeline.steps[1][1].coef_[0]:g}"
            )
    return labels, linear_coeffs
