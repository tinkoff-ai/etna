from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar import DeepARModel
    from etna.models.nn.rnn import RNN
    from etna.models.nn.tft import TFTModel
