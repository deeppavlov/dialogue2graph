import logging

class Logger(object):

    log: logging.Logger
    def __init__(self, name: str, level: int = logging.INFO):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=level,
        )
        self.log = logging.getLogger(name)

    def info(self, *args, **kwargs): # pylint: disable=unused-argument
        self.log.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.log.debug(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.log.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.log.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self.log.critical(*args, **kwargs)
    
    def setLevel(self, level: int):
        self.log.setLevel(level)