import logging
from pythonjsonlogger import jsonlogger
import colorlog
from logging.handlers import (
    RotatingFileHandler,
    TimedRotatingFileHandler
)
import traceback
import sys
from typing import List

class LevelFilter(logging.Filter):
    """
    LevelFilter can be used to implement a wide range of logging policies, such as:
        - Including only messages from certain parts of your application
        - Including only messages of a certain severity
        - Including only messages that contain certain text
    """
    replace_lvl = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, levels: List[str]) -> None:
        # Set levels
        self.levels = [
            lvl.replace(self.replace_lvl) for lvl in levels
        ]

    def filter(self, record) -> bool:
        # Filter levels
        if record.levelno in self.levels:
            return True

def get_logger(
    name: str = None, 
    level: str = None,
    txt_fmt: str = None,
    json_fmt: str = None,
    filter_lvls: List[str] = None,
    log_file: str = None,
    backup_count: int = None
) -> logging.Logger:
    # Define default name, level & formats
    if name is None:
        name = __name__
    if level is None:
        level = logging.INFO
    if json_fmt is None:
        json_fmt = "%(name)s %(asctime)s %(levelname)s %(filename)s %(lineno)s %(message)s"
    if txt_fmt is None:
        txt_fmt = "%(name)s: %(white)s%(asctime)s%(reset)s | %(log_color)s%(levelname)s%(reset)s | %(blue)s%(filename)s:%(lineno)s%(reset)s | %(log_color)s%(message)s%(reset)s"

    # Interpret level
    interpreted_levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    if level in interpreted_levels.keys():
        level = interpreted_levels.get(level)

    # Instanciate the logger
    logging.getLogger().setLevel(level=level)
    logger = logging.getLogger(name) # .setLevel(level=level)
    logger.setLevel(level=level)

    # Build a ColoredFormatter (text)
    txt_formatter = colorlog.ColoredFormatter(fmt=txt_fmt)

    # Build Filters
    if filter_lvls is not None:
        level_filter = LevelFilter(levels=filter_lvls)
    else:
        level_filter = None

    # Add filters to logger
    if level_filter is not None:
        logger.addFilter(level_filter)

    # Add a StreamHandler to output log messages to the standard output (stdout)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(level=level)
    stdout_handler.setFormatter(txt_formatter)

    if level_filter is not None:
        stdout_handler.addFilter(level_filter)

    logger.addHandler(stdout_handler)

    if log_file is not None:
        """
        FileHandler: vanilla file handler to output log messages to a file
        RotatingFileHandler:
            - The log_file will will be created and written to as before until it reaches 5 megabytes.
            - This process continues until we get to logs.txt.5. At that point, the oldest file (logs.txt.5) 
              gets deleted to make way for the newer logs.
        TimedRotatingFileHandler:
            - Rotate files once a week at Sundays, while keeping a maximum of backup_count backup files
        """
        # Assert that the log_file ends with .log
        assert log_file.endswith('.log')

        # Define log_dir
        log_dir = f'logs/{log_file}'

        # Add a FileHandler to output log messages to a file
        # file_handler = logging.FileHandler(log_file)
        # file_handler = RotatingFileHandler(log_file, backupCount=5, maxBytes=5000000)
        file_handler = TimedRotatingFileHandler(
            filename=log_dir, 
            backupCount=backup_count, 
            when="W6" # Sundays
        )
        file_handler.setLevel(level=level)

        # Build a JsonFormatter to customize the log message format (json)
        json_formatter = jsonlogger.JsonFormatter(
            fmt=json_fmt,
            rename_fields={
                "name": "logger_name",
                "asctime": "timestamp",
                "levelname": "severity", 
            }
        )
        file_handler.setFormatter(json_formatter)

        if level_filter is not None:
            file_handler.addFilter(level_filter)

        logger.addHandler(file_handler)

    # Add logging for uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Get the traceback as a string
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        logger.critical(tb_str, exc_info=(exc_type, exc_value, exc_traceback)) # "Uncaught exception"

    sys.excepthook = handle_exception

    return logger