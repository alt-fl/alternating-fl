import logging

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


def configure_logger(level=logging.DEBUG, filters=None):
    """
    Set the logger's level and filters.

    Parameters:
        level (int): The logging level (e.g., logging.DEBUG, logging.INFO).
        filters (list): A list of logging.Filter instances to add to the logger's handlers.
    """
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)
        if filters:
            # clear existing filters if you need to replace them.
            handler.filters = []
            for filt in filters:
                handler.addFilter(filt)


class MessageContentFilter(logging.Filter):
    def __init__(self, banned_content):
        """
        Initialize the filter.

        Parameters:
            banned_content (str or list): A substring or list of substrings to filter out.
        """
        super().__init__()
        if isinstance(banned_content, str):
            self.banned_contents = [banned_content]
        else:
            self.banned_contents = banned_content

    def filter(self, record):
        # record.getMessage() returns the formatted log message.
        msg = record.getMessage()
        # Return False (filter out) if any banned substring is in the message.
        for banned in self.banned_contents:
            if banned in msg:
                return False
        return True
