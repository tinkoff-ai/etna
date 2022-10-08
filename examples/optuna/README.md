# Optuna TPE hyperparameter tuning example

Define your pipeline and hyperparameters in `optuna_example.py`, in the example we will optimize number of `iterations`, `depth` and number of `lags` for `CatBoostMultiSegmentModel`

Run optimization:

```bash
    python optuna_example.py --n-trials=100 --metric-name=MAE
```

You can use [optuna dashboard](https://github.com/optuna/optuna-dashboard) to visualize the results:

```bash
    optuna-dashboard sqlite:///<storage-name>
```
