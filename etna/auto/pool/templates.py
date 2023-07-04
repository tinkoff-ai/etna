# fmt: off
DEFAULT = [
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.NaiveModel', 'lag': 1}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.NaiveModel', 'lag': 7}},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.MovingAverageModel', 'window': '${mult:${horizon},1}'}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.MovingAverageModel', 'window': '${mult:${horizon},2}'}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.MovingAverageModel', 'window': '${mult:${horizon},3}'}},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.SeasonalMovingAverageModel', 'seasonality': 7, 'window': 3}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.SeasonalMovingAverageModel', 'seasonality': '${horizon}', 'window': 3}},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.HoltWintersModel'}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.HoltWintersModel', 'damped_trend': True, 'seasonal': 'add', 'trend': 'add'}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.HoltWintersModel', 'damped_trend': False, 'seasonal': 'add', 'trend': 'add'}} ,

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.AutoARIMAModel'}},

    # {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.TBATSModel'}},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.LinearPerSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.StandardScalerTransform', 'in_column': 'target'}, {'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}]},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.LinearMultiSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.StandardScalerTransform', 'in_column': 'target', 'mode': 'macro'}, {'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}]},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.ElasticPerSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.StandardScalerTransform', 'in_column': 'target'}, {'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}]},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.ElasticMultiSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.StandardScalerTransform', 'in_column': 'target', 'mode': 'macro'}, {'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}]},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.CatBoostMultiSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.StandardScalerTransform', 'in_column': 'target', 'mode': 'macro'}, {'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}, {'_target_': 'etna.transforms.SegmentEncoderTransform'}, {'_target_': 'etna.transforms.DateFlagsTransform', 'day_number_in_week': True, 'is_weekend': True, 'week_number_in_year': True}]},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.CatBoostMultiSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}, {'_target_': 'etna.transforms.SegmentEncoderTransform'}, {'_target_': 'etna.transforms.DateFlagsTransform', 'day_number_in_week': True, 'is_weekend': True, 'week_number_in_year': True}]},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.CatBoostPerSegmentModel'}, 'transforms': [{'_target_': 'etna.transforms.LagTransform', 'in_column': 'target', 'lags': '${shift:${horizon},[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}'}, {'_target_': 'etna.transforms.DateFlagsTransform', 'day_number_in_week': True, 'is_weekend': True, 'week_number_in_year': True}]},

    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.ProphetModel', 'seasonality_mode': 'multiplicative'}},
    {'_target_': 'etna.pipeline.Pipeline', 'horizon': '${__aux__.horizon}', 'model': {'_target_': 'etna.models.ProphetModel', 'seasonality_mode': 'additive'}},
]
# fmt: on
