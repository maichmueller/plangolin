import datetime
import logging
import logging.handlers
from functools import cache
from io import StringIO

from colorama import Fore, Style, init
from tqdm.auto import tqdm as tqdm_

# init colorama for cross-platform color support
init(autoreset=True)

LOG_COLORS = {
    "INFO": Fore.GREEN,
    "DEBUG": Fore.BLUE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}


class SPDLOGFormatter(logging.Formatter):
    """
    Custom log formatter with color support and custom log to match SPDLOG's format
    """

    spdlog_format_string = (
        "[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
    )

    def __init__(self):
        super().__init__(fmt=self.spdlog_format_string, datefmt="%Y-%m-%d %H:%M:%S")

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname.lower()}{Style.RESET_ALL}"
        return super().format(record)


class TqdmLogFormatter(SPDLOGFormatter):
    def format(self, record):
        # Get the original message.
        original_msg = record.getMessage()
        if original_msg == "\n":
            return original_msg
        cr_prefix = ""
        if original_msg.startswith("\r"):
            # If there are leading '\r' characters, preserve them.
            # (You might want to preserve just one or all; here we capture all leading '\r's.)
            num_cr = len(original_msg) - len(original_msg.lstrip("\r"))
            cr_prefix = "\r" * num_cr
            # Remove the leading carriage returns from the message attribute for formatting.
            record.msg = original_msg.lstrip("\r")
            # Note: record.args remains unchanged

        # Format the message as usual.
        formatted = super().format(record)
        # Prepend the captured carriage return(s) to the final formatted string.
        if cr_prefix:
            formatted = cr_prefix + formatted

        return formatted


class TqdmLogGuard:
    def __init__(self, logger):
        self._logger = logger

    def __enter__(self):
        self.__original_formatters = list()

        for handler in self._logger.handlers:
            self.__original_formatters.append(handler.formatter)

            handler.terminator = ""
            formatter = TqdmLogFormatter()
            handler.setFormatter(formatter)

        return self._logger

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handler, formatter in zip(
            self._logger.handlers, self.__original_formatters
        ):
            handler.terminator = "\n"
            handler.setFormatter(formatter)


class TqdmLogger(StringIO):
    def __init__(self, logger):
        super().__init__()

        self._logger = logger

    def write(self, buffer):
        with TqdmLogGuard(self._logger) as logger:
            logger.info(buffer)

    def flush(self):
        pass


def setup_logger(name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(SPDLOGFormatter())
    logger.addHandler(console_handler)
    return logger


@cache
def get_logger(name="root"):
    logger = logging.getLogger(name)
    setup_logger(name)
    return logger


def tqdm(x=None, **log_kwargs):
    logger = log_kwargs.pop("logger", logging.getLogger("root"))
    ascii_ = log_kwargs.pop("ascii", False)
    return tqdm_(
        x,
        file=TqdmLogger(logger),
        ascii=ascii_,
        **log_kwargs,
    )
