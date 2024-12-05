import logging

def setup_logger(name, log_file=None, level=logging.INFO):
    """Function to set up a logger."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # If a log file is specified, log to both file and console
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers = [handler, file_handler]
    else:
        handlers = [handler]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    for h in handlers:
        logger.addHandler(h)

    return logger