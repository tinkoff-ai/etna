<h1 align="center">ETNA Time Series Library</h1>
<h3 align="center">Predict your time series the easiest way</h3>

<p align="center">
  <a href="https://pypi.org/project/etna/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/etna.svg" /></a>
  <a href="https://pypi.org/project/etna/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/etna.svg" /></a>
  <a href="https://pepy.tech/project/etna"><img alt="Downloads" src="https://static.pepy.tech/personalized-badge/etna?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads" /></a>
</p>
 
<p align="center">
  <a href="https://codecov.io/gh/tinkoff-ai/etna"><img alt="Coverage" src="https://img.shields.io/codecov/c/github/tinkoff-ai/etna.svg" /></a>
  <a href="https://github.com/tinkoff-ai/etna/actions/workflows/test.yml?query=branch%3Amaster++"><img alt="Test passing" src="https://img.shields.io/github/workflow/status/tinkoff-ai/etna/Test/master?label=tests" /></a>
  <a href="https://github.com/tinkoff-ai/etna/actions/workflows/publish.yml"><img alt="Docs publish" src="https://img.shields.io/github/workflow/status/tinkoff-ai/etna/Publish?label=docs" /></a>
  <a href="https://github.com/tinkoff-ai/etna/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/tinkoff-ai/etna.svg" /></a>
</p>

<p align="center">
  <a href="https://t.me/etna_support"><img alt="Telegram" src="https://img.shields.io/badge/channel-telegram-blue" /></a>
  <a href="https://opendatascience.slack.com/archives/C02Q62NEPH8"><img alt="Slack" src="https://img.shields.io/badge/slack-ods.ai-orange" /></a>
  <a href="https://github.com/tinkoff-ai/etna/discussions"><img alt="GitHub Discussions" src="https://img.shields.io/github/discussions/tinkoff-ai/etna" /></a>
  <a href="https://github.com/tinkoff-ai/etna/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/tinkoff-ai/etna.svg" /></a>
  <a href="https://github.com/tinkoff-ai/etna/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/tinkoff-ai/etna?style=social" /></a>
</p>

[Homepage](https://etna.tinkoff.ru) |
[Documentation](https://etna-docs.netlify.app/) |
[Tutorials](https://github.com/tinkoff-ai/etna/tree/master/examples) | 
[Contribution Guide](https://github.com/tinkoff-ai/etna/blob/master/CONTRIBUTING.md) |
[Release Notes](https://github.com/tinkoff-ai/etna/releases)

  
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
Contributions are welcome - check our [Contribution Guide](https://github.com/tinkoff-ai/etna/blob/master/CONTRIBUTING.md).



## Installation 

ETNA is on [PyPI](https://pypi.org/project/etna), so you can use `pip` to install it.

```bash
pip install --upgrade pip
pip install etna
```


## Get started 
Here's some example code for a quick start.
```python
import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel
from etna.pipeline import Pipeline

# Read the data
df = pd.read_csv("examples/data/example_dataset.csv")

# Create a TSDataset
df = TSDataset.to_dataset(df)
ts = TSDataset(df, freq="D")

# Choose a horizon
HORIZON = 8

# Fit the pipeline
pipeline = Pipeline(model=ProphetModel(), horizon=HORIZON)
pipeline.fit(ts)

# Make the forecast
forecast_ts = pipeline.forecast()
```

## Tutorials
We have also prepared a set of tutorials for an easy introduction:

| Notebook     | Interactive launch  |
|:----------|------:|
| [Get started](https://github.com/tinkoff-ai/etna/tree/master/examples/get_started.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/get_started.ipynb) |
| [Backtest](https://github.com/tinkoff-ai/etna/tree/master/examples/backtest.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/backtest.ipynb) |
| [EDA](https://github.com/tinkoff-ai/etna/tree/master/examples/EDA.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/EDA.ipynb) |
| [Outliers](https://github.com/tinkoff-ai/etna/tree/master/examples/outliers.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/outliers.ipynb) |
| [Clustering](https://github.com/tinkoff-ai/etna/tree/master/examples/clustering.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/clustering.ipynb) |
| [Deep learning models](https://github.com/tinkoff-ai/etna/tree/master/examples/NN_examples.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/NN_examples.ipynb) |
| [Ensembles](https://github.com/tinkoff-ai/etna/tree/master/examples/ensembles.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tinkoff-ai/etna/master?filepath=examples/ensembles.ipynb) |

## Documentation
ETNA documentation is available [here](https://etna-docs.netlify.app/).

## Resources

- [Forecasting with ETNA: Fast and Furious](https://medium.com/its-tinkoff/forecasting-with-etna-fast-and-furious-1b58e1453809) on Medium

- [Store sales prediction with etna library](https://www.kaggle.com/dmitrybunin/store-sales-prediction-with-etna-library?scriptVersionId=81104235) on Kaggle

- [PyCon Russia September 2021 talk](https://youtu.be/VxWHLEFgXnE) on YouTube

## Acknowledgments

### ETNA.Team
[Andrey Alekseev](https://github.com/iKintosh),
[Nikita Barinov](https://github.com/diadorer),
[Dmitriy Bunin](https://github.com/Mr-Geekman),
[Aleksandr Chikov](https://github.com/alex-hse-repository),
[Vladislav Denisov](https://github.com/v-v-denisov),
[Martin Gabdushev](https://github.com/martins0n),
[Sergey Kolesnikov](https://github.com/Scitator),
[Artem Makhin](https://github.com/Ama16),
[Ivan Mitskovets](https://github.com/imitskovets),
[Albina Munirova](https://github.com/albinamunirova),
[Nikolay Romantsov](https://github.com/WinstonDovlatov),
[Julia Shenshina](https://github.com/julia-shenshina)

### ETNA.Contributors
[Artem Levashov](https://github.com/soft1q),
[Aleksey Podkidyshev](https://github.com/alekseyen),
[Carlosbogo](https://github.com/Carlosbogo)

## License

Feel free to use our library in your commercial and private applications.

ETNA is covered by [Apache 2.0](/LICENSE). 
Read more about this license [here](https://choosealicense.com/licenses/apache-2.0/)
