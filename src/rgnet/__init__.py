import spdlog

# initialize the logger of the main module
try:
    logger = spdlog.ConsoleLogger("default")
    logger.set_level(spdlog.LogLevel.INFO)
except RuntimeError:
    # logger already exists
    pass

from .encoding import *
from .models import *
from .rl import *
from .supervised import *
