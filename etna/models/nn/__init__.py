from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.deepar import DeepARBaseEtnaModel
    from etna.models.nn.tft import TFTBaseEtnaModel
