import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s:%(name)s:%(message)s",
    "%H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
