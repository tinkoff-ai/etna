def _test_prediction_decomposition(model, train, test, prediction_size=None):
    model.fit(train)

    predict_args = dict(ts=train, return_components=True)
    forecast_args = dict(ts=test, return_components=True)

    if prediction_size is not None:
        predict_args["prediction_size"] = prediction_size
        forecast_args["prediction_size"] = prediction_size

    forecast = model.predict(**predict_args)
    assert len(forecast.target_components_names) > 0

    forecast = model.forecast(**forecast_args)
    assert len(forecast.target_components_names) > 0
