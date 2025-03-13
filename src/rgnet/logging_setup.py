import datetime
import logging
import logging.handlers

from colorama import Fore, Style, init

# init colorama for cross-platform color support
init(autoreset=True)

LOG_COLORS = {
    "INFO": Fore.GREEN,
    "DEBUG": Fore.BLUE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter with color support and custom log to match SPDLOG's format
    """

    format_string = (
        "[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s] %(message)s"
    )

    def __init__(self):
        super().__init__(fmt=self.format_string, datefmt="%Y-%m-%d %H:%M:%S")

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname.lower()}{Style.RESET_ALL}"
        return super().format(record)


def setup_logger(name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    return logger
