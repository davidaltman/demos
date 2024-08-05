""" defines a logger object providing filename, line number, and timestamp to output
"""
import logging


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s\
    [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger(name)
    return logger
