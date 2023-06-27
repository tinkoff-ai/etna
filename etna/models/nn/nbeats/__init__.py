from etna import SETTINGS

if SETTINGS.torch_required:
    from etna.models.nn.nbeats.nbeats import NBeatsGenericModel
    from etna.models.nn.nbeats.nbeats import NBeatsInterpretableModel
