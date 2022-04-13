# Stacking investigation

We have a concern that our stacking ensemble doesn't work correctly.

To check different results an example from README was taken as a base and different models were evaluated in backtest with 5 folds.

## Running

To get results run `main.py`.

## Raw results

* Naive-1: {'MAE': 82.275, 'SMAPE': 18.344640953947827}
* Naive-7: {'MAE': 32.24642857142857, 'SMAPE': 7.292286359988653}
* CatBoost: {'MAE': 26.940758936571264, 'SMAPE': 6.0450193198991515}
* Prophet: {'MAE': 29.580124555312285, 'SMAPE': 6.996532518059524}
* SARIMAX: {'MAE': 34.63934027388464, 'SMAPE': 8.843575996117703}
* HoltWinters: {'MAE': 25.187777109658114, 'SMAPE': 5.956625636968684}

Stacking (n_folds=1):
* LinearRegression(): {'MAE': 43.117206619212624, 'SMAPE': 11.87798041968703}
* Ridge(): {'MAE': 43.104284408303684, 'SMAPE': 11.873984050150831}
* Ridge(positive=True): {'MAE': 30.22162468204244, 'SMAPE': 7.342776317795026}

Stacking (n_folds=3):
* LinearRegression(): {'MAE': 30.338531332956563, 'SMAPE': 7.498824773506746}
* Ridge(): {'MAE': 30.338335288429906, 'SMAPE': 7.498750956032133}
* Ridge(positive=True): {'MAE': 28.257038243760938, 'SMAPE': 6.306203678099658}
* Lasso(): {'MAE': 31.10874812494723, 'SMAPE': 7.803452055027678}
* Lasso(positive=True): {'MAE': 27.71845107005554, 'SMAPE': 6.228994824727682}
* CatBoostRegressor(): {'MAE': 30.686572149841894, 'SMAPE': 6.922554400585119}

Stacking (n_folds=5):
* Linear(): {'MAE': 28.396750751464715, 'SMAPE': 6.7673855758620345}
* Ridge(): {'MAE': 28.39669937783745, 'SMAPE': 6.767362209630145}
* Ridge(positive=True): {'MAE': 28.544542995345697, 'SMAPE': 6.405880742348323}

Voting (n_folds=1):
* weights=None: {'MAE': 30.596823149626772, 'SMAPE': 6.936477635728394}
* weights=auto: {'MAE': 28.451714621236587, 'SMAPE': 6.398404134157372}

Voting (n_folds=3):
* weights=None: {'MAE': 30.596823149626772, 'SMAPE': 6.936477635728394}
* weights=auto: {'MAE': 27.316527822817903, 'SMAPE': 6.285625605003396}

## Conclusions

1. Unfortunately, ensembles don't achieve as high results as best single models.
2. Small amount of data (`n_folds=1`) hurts the performance
3. After some threshold for amount of data (`n_folds=3`) we don't see improvements of quality
4. It looks like `Ridge(positive=True)` and `Lasso(positive=True)` perform better then their analogous without restriction to weights. It can be reasonalbe to change our default model for stacking.
