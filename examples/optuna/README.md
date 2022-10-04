# Optuna TPE hyperparameter tuning example

Define your pipeline and hyperparameters in `optuna_example.py`, in the example we will optimize number of iterations `iterations` and `depth` and number of `lags`.

Run optimization:

```bash
    python optuna_example.py --n-trials=100 --metric-name=MAE
```
