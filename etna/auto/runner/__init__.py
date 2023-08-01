from etna import SETTINGS

if SETTINGS.auto_required:
    from etna.auto.runner.base import AbstractRunner
    from etna.auto.runner.local import LocalRunner
    from etna.auto.runner.local import ParallelLocalRunner
