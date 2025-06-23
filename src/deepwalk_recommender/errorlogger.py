"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 02:53:50
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:10:28
FilePath: src/deepwalk_recommender/errorlogger.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 02:53:50
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-23 23:10:28
FilePath: src/deepwalk_recommender/errorlogger.py
Description: Structured logging utility for recommendation engine
"""

import inspect
import sys
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from src.deepwalk_recommender.config import PathConfig


class LogLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Logger:
    """
    Structured logger for the DeepWalk recommender system, supporting info, warning, error, and debug messages.
    Logs are formatted with timestamps, function context, and optional error tracebacks, and written to organization-specific log files.
    Provides methods to clear logs and prevent duplicate log entries.
    """

    def __init__(self, log_dir: str = PathConfig.LOG_DIR):
        """
        Initializes the Logger instance by setting up log directory paths, log file mappings for each log level, ensuring the log directory exists, and initializing the log cache to prevent duplicate entries.

        Args:
            log_dir (str): Path to the directory where log files will be stored. Defaults to PathConfig.LOG_DIR.
        """
        self.log_dir = Path(log_dir)
        self.log_files = {
            LogLevel.INFO: PathConfig.INFO_LOG,
            LogLevel.WARNING: PathConfig.WARNING_LOG,
            LogLevel.ERROR: PathConfig.ERROR_LOG,
        }
        self._ensure_log_directory()
        self._log_cache: set[int] = set()

    def _ensure_log_directory(self) -> None:
        """
        Ensure that the log directory exists by creating it and any necessary parent directories if they do not already exist.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_caller_info() -> tuple[str, str]:
        """
        Retrieve the names of the current and parent calling functions from the call stack.

        Returns:
            tuple[str, str]: A tuple containing the current function name and its parent function name.
        """
        stack = inspect.stack()
        caller_frame = next(
            (frame for frame in stack if frame.filename != __file__),
            stack[2] if len(stack) > 2 else None,
        )
        current_function = caller_frame.function if caller_frame else "Unknown"
        parent_function = stack[3].function if len(stack) > 3 else "Unknown"
        return current_function, parent_function

    def _format_message(
        self,
        level: LogLevel,
        message: str,
        error: Optional[Exception] = None,
        additional_info: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> str:
        """
        Format a structured log message with timestamp, log level, function context, message, optional error details, traceback, and additional context.

        Args:
            level (LogLevel): The severity level of the log.
            message (str): The main log message.
            error (Optional[Exception], optional): Exception instance to include error details. Defaults to None.
            additional_info (Optional[Dict[str, Any]], optional): Extra context to include in the log. Defaults to None.
            exc_info (bool, optional): Whether to include the full traceback if an error is provided. Defaults to False.

        Returns:
            str: The formatted log message as a string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_function, parent_function = self._get_caller_info()

        lines = [
            "=" * 80,
            f"TIMESTAMP: {timestamp}",
            f"LEVEL: {level.value}",
            f"FUNCTION: {current_function}",
            f"PARENT FUNCTION: {parent_function}",
            "-" * 80,
            f"MESSAGE: {message}",
        ]

        if error:
            lines.extend(
                [
                    f"ERROR TYPE: {type(error).__name__}",
                    f"ERROR MESSAGE: {str(error)}",
                    "-" * 80,
                ]
            )
            if exc_info:
                try:
                    tb = getattr(error, "__traceback__", None)
                    if tb:
                        trace_lines = traceback.format_exception(type(error), error, tb)
                    else:
                        trace_lines = traceback.format_exc().splitlines()
                    lines.append("FULL TRACEBACK:")
                    lines.extend(trace_lines)
                    lines.append("-" * 80)
                except Exception as format_err:
                    lines.append(f"Failed to format traceback: {format_err}")

        context = {"ai_engineer": "Muhammad"}
        if additional_info:
            context.update(additional_info)

        lines.extend(
            [
                "CONTEXT:",
                "\n".join(f"{k}: {v}" for k, v in context.items()),
                "=" * 80 + "\n",
            ]
        )

        return "\n".join(lines)

    def _write_log(self, level: LogLevel, message: str) -> None:
        """
        Write a formatted log message to the appropriate log file for the given log level, ensuring the log directory exists and preventing duplicate log entries. Handles file I/O errors gracefully.
        """
        log_hash = hash(message)
        if log_hash in self._log_cache:
            return  # prevent duplicate writes

        try:
            self._ensure_log_directory()
            with open(self.log_files[level], "a", encoding="utf-8") as f:
                f.write(message)
            self._log_cache.add(log_hash)
        except (IOError, PermissionError) as e:
            print(f"Failed to write log: {e}", file=sys.stderr)

    def info(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an informational message to the info log file with optional additional context.

        Args:
            message (str): The message to log.
            additional_info (Optional[Dict[str, Any]]): Extra context to include in the log entry.
        """
        formatted = self._format_message(
            LogLevel.INFO, message, additional_info=additional_info
        )
        self._write_log(LogLevel.INFO, formatted)

    def warning(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a warning message to the warning log file with optional additional context.

        Args:
            message (str): The warning message to log.
            additional_info (Optional[Dict[str, Any]]): Extra context to include in the log entry.
        """
        formatted = self._format_message(
            LogLevel.WARNING, message, additional_info=additional_info
        )
        self._write_log(LogLevel.WARNING, formatted)

    def error(
        self,
        error: Exception,
        additional_info: Optional[Dict[str, Any]] = None,
        exc_info: bool = True,
    ) -> None:
        """
        Log an error message with structured formatting, including exception details, optional traceback, and additional context, to the error log file.

        Args:
            error (Exception): The exception instance to log.
            additional_info (Optional[Dict[str, Any]]): Extra context to include in the log entry.
            exc_info (bool): Whether to include the full traceback. Defaults to True.
        """
        formatted = self._format_message(
            LogLevel.ERROR,
            "An error occurred",
            error=error,
            additional_info=additional_info,
            exc_info=exc_info,
        )
        self._write_log(LogLevel.ERROR, formatted)

    def debug(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a debug-level message by prefixing it with '[DEBUG]' and writing it as an info log entry, with optional additional context.

        Args:
            message (str): The debug message to log.
            additional_info (Optional[Dict[str, Any]]): Extra context to include in the log entry.
        """
        self.info(f"[DEBUG] {message}", additional_info=additional_info)

    def clear_logs(self, level: Optional[LogLevel] = None) -> None:
        """
        Clear the contents of log files for the specified log level or all levels, and reset the log cache.

        Args:
            level (Optional[LogLevel]): The log level to clear. If None, clears all log files.
        """
        try:
            targets = [self.log_files[level]] if level else self.log_files.values()
            for file in targets:
                with open(file, "w", encoding="utf-8") as f:
                    f.write("")
            self._log_cache.clear()
        except Exception as e:
            print(f"Failed to clear logs: {e}", file=sys.stderr)


# Main logger instance
system_logger = Logger()


# Helper to log structured errors from anywhere
def log_error(message: str, **kwargs):
    """
    Log a structured error message without raising an exception.

    Args:
        message (str): The error message to log.
        **kwargs: Additional context to include in the log entry.
    """
    formatted = system_logger._format_message(
        level=LogLevel.ERROR, message=message, additional_info=kwargs, exc_info=False
    )
    system_logger._write_log(LogLevel.ERROR, formatted)
