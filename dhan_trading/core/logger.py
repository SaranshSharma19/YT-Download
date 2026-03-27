import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    file_path = log_dir / log_file
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler with rotation
    file_handler = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Common loggers
system_logger = setup_logger('system', 'trading_system.log')
trade_logger = setup_logger('trades', 'trades.log')
error_logger = setup_logger('errors', 'errors.log', level=logging.ERROR)
