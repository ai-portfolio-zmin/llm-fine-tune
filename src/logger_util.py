import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)
    return logger