""" Logger class
"""
import logging


class Logger(object):
    """Logger class to simplify logging operations"""

    __slots__ = ["log"]
    log: logging.Logger

    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize logger with given name and level.

        Args:
          name: name of the logger
          level: logging level
        """
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=level,
        )
        self.log = logging.getLogger(name)

    def info(self, *args, **kwargs):  # pylint: disable=unused-argument
        """
        Log information message.

        Args:
          *args: log message
          **kwargs: additional arguments for logging
        """
        self.log.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        """
        Log debug message.

        Args:
          *args: log message
          **kwargs: additional arguments for logging
        """
        self.log.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """
        Log warning message.

        Args:
          *args: log message
          **kwargs: additional arguments for logging
        """
        self.log.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """
        Log error message.

        Args:
          *args: log message
          **kwargs: additional arguments for logging
        """
        self.log.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        """
        Log critical message.

        Args:
          *args: log message
          **kwargs: additional arguments for logging
        """
        self.log.critical(*args, **kwargs)

    def setLevel(self, level: int):
        """
        Set the logging level for the logger.

        Args:
          level: logging level
        """
        self.log.setLevel(level)
