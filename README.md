# ETNA Time Series Library

[![Pipi version](https://img.shields.io/pypi/v/etna-ts.svg)](https://pypi.org/project/etna-ts/)
[![PyPI Status](https://static.pepy.tech/personalized-badge/etna-ts?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/etna-ts)
[![Coverage](https://img.shields.io/codecov/c/github/tinkoff-ai/etna-ts)](https://codecov.io/gh/tinkoff-ai/etna-ts)

[![Telegram](https://img.shields.io/badge/channel-telegram-blue)](https://t.me/etna_support)

[Homepage](https://etna.tinkoff.ru) |
[Documentation](https://etna-docs.netlify.app/) |
[Tutorials](https://github.com/tinkoff-ai/etna-ts/tree/master/examples) | 
[Contribution Guide](https://github.com/tinkoff-ai/etna-ts/blob/master/CONTRIBUTING.md) |
[Release Notes](https://github.com/tinkoff-ai/etna-ts/releases)

ETNA is an easy-to-use time series forecasting framework. 
It includes built in toolkits for time series preprocessing, feature generation, 
a variety of predictive models with unified interface - from classic machine learning
to SOTA neural networks, models combination methods and smart backtesting.
ETNA is designed to make working with time series simple, productive, and fun. 

ETNA is the first python open source framework of 
[Tinkoff.ru](https://www.tinkoff.ru/eng/)
Artificial Intelligence Center. 
The library started as an internal product in our company - 
we use it in over 10+ projects now, so we often release updates. 
Contributions are welcome - check our [Contribution Guide](https://github.com/tinkoff-ai/etna-ts/blob/master/CONTRIBUTING.md).



## Installation 

ETNA is on [PyPI](https://pypi.org/project/etna-ts), so you can use `pip` to install it.

```bash
pip install --upgrade pip
pip install etna-ts
```


## Get started 
Here's some example code for a quick start.
```python
import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel

# Read the data
df = pd.read_csv("examples/data/example_dataset.csv")

# Create a TSDataset
df = TSDataset.to_dataset(df)
ts = TSDataset(df, freq="D")

# Choose a horizon
HORIZON = 8

# Fit the model
model = ProphetModel()
model.fit(ts)

# Make the forecast
future_ts = ts.make_future(HORIZON)
forecast_ts = model.forecast(future_ts)
```

## Tutorials
We have also prepared a set of tutorials for an easy introduction:

| Notebook     | Interactive launch  |
|:----------|------:|
| [Get started](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/get_started.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/get_started.ipynb) |
| [Backtest](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/backtest.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/backtest.ipynb) |
| [EDA](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/EDA.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/EDA.ipynb) |
| [Outliers](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/outliers.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/outliers.ipynb) |
| [Clustering](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/clustering.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/clustering.ipynb) |
| [Deep learning models](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/NN_examples.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/NN_examples.ipynb) |
| [Ensembles](https://github.com/tinkoff-ai/etna-ts/tree/master/examples/ensembles.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna-ts/master?filepath=examples/ensembles.ipynb) |

## Documentation
ETNA documentation is available [here](https://etna-docs.netlify.app/).

## Acknowledgments

### ETNA.Team
[Alekseev Andrey](https://github.com/iKintosh), 
[Shenshina Julia](https://github.com/julia-shenshina),
[Gabdushev Martin](https://github.com/martins0n),
[Kolesnikov Sergey](https://github.com/Scitator),
[Bunin Dmitriy](https://github.com/Mr-Geekman),
[Chikov Aleksandr](https://github.com/alex-hse-repository),
[Barinov Nikita](https://github.com/diadorer),
[Romantsov Nikolay](https://github.com/WinstonDovlatov),
[Makhin Artem](https://github.com/Ama16),
[Denisov Vladislav](https://github.com/v-v-denisov),
[Mitskovets Ivan](https://github.com/imitskovets),
[Munirova Albina](https://github.com/albinamunirova)


### ETNA.Contributors
[Levashov Artem](https://github.com/soft1q),
[Podkidyshev Aleksey](https://github.com/alekseyen)

## License

Feel free to use our library in your commercial and private applications.

ETNA is covered by [Apache 2.0](/LICENSE). 
Read more about this license [here](https://choosealicense.com/licenses/apache-2.0/)
