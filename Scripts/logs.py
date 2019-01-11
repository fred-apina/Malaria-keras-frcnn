from imports import *


def log_results(file_name):

    file=file="..Logs/{}.log".format(file_name)

    if not os.path.isfile(file):
        open(file, "w+").close()

    console_logging_format = '%(levelname)s %(message)s'
    file_logging_format = '%(levelname)s: %(asctime)s: %(message)s'

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()
    # create a file handler for output file
    handler = logging.FileHandler(file)

    # set the logging level for log file
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    if os.stat(file).st_size == 0:
        logger.info("[epoch:tra_loss:val_loss:tra_f1:val_f1:tra_best_f1:val_best_f1]")

    return logger



    