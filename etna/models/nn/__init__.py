from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar import DeepARModel
    from etna.models.nn.mlp import MLPModel
    from etna.models.nn.rnn import RNNModel
    from etna.models.nn.tft import TFTModel
