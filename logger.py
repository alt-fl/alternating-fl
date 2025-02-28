import logging
import sys

logging.captureWarnings(True)
# create a logger with a specific name, and it's console handler
logger = logging.getLogger("Alt-FL")
logger.propagate = False

console_handler = logging.StreamHandler()

# create a formatter that includes the time, logger name, level, and message.
formatter = logging.Formatter(
    fmt="[%(name)s|%(levelname)s|%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# add the handler to the logger if it doesn't already have one
if not logger.handlers:
    logger.addHandler(console_handler)

# set a default logging level.
logger.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)

# forces warnings to be formatted according the my formatter
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.propagate = False
warnings_logger.handlers = []
warnings_logger.addHandler(console_handler)
warnings_logger.setLevel(logging.WARNING)


def configure_logger(level=logging.DEBUG, filters=None, override_stdout=True):
    """
    Set the logger's level and filters.

    Parameters:
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        filters (list): A list of logging.Filter instances to add to the logger's handlers.
    """
    logger.setLevel(level)
    if override_stdout:
        sys.stdout = LoggerWriter(logger, level)

    for handler in logger.handlers:
        handler.setLevel(level)
        if filters:
            # clear existing filters if you need to replace them.
            handler.filters = []
            for filt in filters:
                handler.addFilter(filt)


class LoggerWriter:
    """
    A file-like object that redirects writes to a logger.

    Not a very elegant solution for filtering messages, but it will have to do
    with print statemtns to stdout...
    """

    def __init__(self, logger, level=logging.DEBUG):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # buffer the message to handle partial writes
        self._buffer += message
        # if the buffer contains one or more newline, log each line
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():  # avoid logging empty lines
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        # flush any remaining message in the buffer
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""


class MessageContentFilter(logging.Filter):
    def __init__(self, banned_content):
        super().__init__()
        # support a single string or a list of banned substrings.
        if isinstance(banned_content, str):
            self.banned_contents = [banned_content]
        else:
            self.banned_contents = banned_content

    def filter(self, record):
        message = record.getMessage()
        # return False if any banned substring is found.
        return not any(banned in message for banned in self.banned_contents)
