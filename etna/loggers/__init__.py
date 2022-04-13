"""Module with loggers.

Implementation of global logger ``tslogger`` is inspired by loguru.

Examples
--------
>>> from etna.loggers import tslogger, ConsoleLogger
>>> tslogger.add(ConsoleLogger())
0

Notes
-----
Global objects behavior could be different while parallel usage because platform dependent new process start.
Be sure that new process is started with ``fork``.
If it's not possible you should try define all globals before ``if __name__ == "__main__"`` scope.
"""
from etna import SETTINGS
from etna.loggers.base import _Logger
from etna.loggers.console_logger import ConsoleLogger
from etna.loggers.file_logger import LocalFileLogger
from etna.loggers.file_logger import S3FileLogger

if SETTINGS.wandb_required:
    from etna.loggers.wandb_logger import WandbLogger

tslogger = _Logger()
