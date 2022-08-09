from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar import DeepARModel
    from etna.models.nn.deepstate.state_space_model import CompositeSSM
    from etna.models.nn.deepstate.state_space_model import LevelSSM
    from etna.models.nn.deepstate.state_space_model import LevelTrendSSM
    from etna.models.nn.deepstate.state_space_model import SeasonalitySSM
    from etna.models.nn.rnn import RNNModel
    from etna.models.nn.tft import TFTModel
