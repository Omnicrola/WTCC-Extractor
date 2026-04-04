"""
Shared logging setup for the WTCC extraction pipeline.

Creates a logger that writes to both the console (plain message, no timestamp)
and a timestamped file in LOGS_DIR (millisecond-accurate timestamps).

Log filename format: YYYY-MM-DD_HH-MM-SS_<name>.log
Log entry format:    [YYYY-MM-DD HH:MM:SS.mmm] message
"""

import logging
import os
from datetime import datetime


class _MsFormatter(logging.Formatter):
    """Logging formatter with millisecond-accurate timestamps."""

    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(record.msecs):03d}"


def setup_logger(name: str, logs_dir: str) -> logging.Logger:
    """
    Return a named logger with two handlers:
      - StreamHandler  : plain message only  (matches existing print behaviour)
      - FileHandler    : [timestamp.ms] message  written to logs_dir

    Safe to call multiple times with the same name — duplicate handlers are
    not added (handles importlib reloading scripts across a batch run).
    """
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't double-log through the root logger

    # Console — plain message only
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File — millisecond timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"{timestamp}_{name}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_MsFormatter("[%(asctime)s] %(message)s"))
    logger.addHandler(fh)

    logger.info(f"=== Log started: {log_path} ===")
    return logger
