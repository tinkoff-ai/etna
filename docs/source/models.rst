Models
======

Models are used to make predictions. Let's look at the basic example of usage:

>>> import pandas as pd
>>> from etna.datasets import TSDataset, generate_ar_df
>>> from etna.transforms import LagTransform
>>> from etna.models import LinearPerSegmentModel
>>>
>>> df = generate_ar_df(periods=100, start_time="2021-01-01", ar_coef=[1/2], n_segments=2)
>>> ts = TSDataset(TSDataset.to_dataset(df), "D")
>>> lag_transform = LagTransform(in_column="target", lags=[3, 4, 5])
>>> ts.fit_transform(transforms=[lag_transform])
>>> future_ts = ts.make_future(3)
>>> model = LinearPerSegmentModel()
>>> model.fit(ts)
LinearPerSegmentModel(fit_intercept = True, normalize = False, )
>>> forecast_ts = model.forecast(future_ts)
segment                 segment_0  ... segment_1
feature    regressor_target_lag_3  ...    target
timestamp                          ...
2021-04-11              -0.090673  ...  0.286764
2021-04-12              -0.665337  ...  0.295589
2021-04-13               0.365363  ...  0.374554
[3 rows x 8 columns]

There is a key note to mention: `future_ts` and `forecast_ts` are the same objects.
Method `forecast` only fills 'target' column in `future_ts` and return reference to it.

>>> forecast_ts is future_ts
True


.. _models:

.. currentmodule:: etna

Details and available models
-------------------------------

See the API documentation for further details on available models:

.. currentmodule:: etna

.. moduleautosummary::
   :toctree: api/
   :template: custom-module-template.rst

   etna.models