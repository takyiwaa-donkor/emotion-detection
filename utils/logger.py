"""
Logging configuration module for the Emotion Detection System.

This module sets up application-wide logging using Python's built-in
logging library. Logs are written to 'app.log' and include timestamps,
log levels, and descriptive messages.

Usage:
    from utils.logger import logger
    logger.info("Emotion detection started.")
"""

import logging

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()