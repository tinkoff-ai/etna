from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar import DeepARModel
    from etna.models.nn.deepstate import DeepStateNetwork
    from etna.models.nn.deepstate import LevelSSM
    from etna.models.nn.deepstate import LevelTrendSSM
    from etna.models.nn.deepstate import SeasonalitySSM
    from etna.models.nn.tft import TFTModel
