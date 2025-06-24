"""
Author: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
Date: 2025-06-22 02:53:50
LastEditors: Muhammad Abiodun SULAIMAN abiodun.msulaiman@gmail.com
LastEditTime: 2025-06-24 04:29:03
FilePath: src/deepwalk_recommender/error_logger.py
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
    """
    Defines logging severity levels with associated log files.

    Levels:
        INFO: For informational messages about system operation
        WARNING: For potentially problematic situations
        ERROR: For serious failures that prevent normal operation

    Each level is mapped to a dedicated log file path.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class Logger:
    """
    Structured logging utility for the recommendation system.

    Features:
    - Writes to level-specific log files (info.log, warning.log, error.log)
    - Prevents duplicate log entries
    - Captures function call context
    - Formats errors with optional tracebacks
    - Includes custom contextual information
    - Provides log clearing functionality

    Args:
        log_dir (str | Path): Directory to store log files (default: PathConfig.LOG_DIR)
    """

    def __init__(self, log_dir: str | Path = PathConfig.LOG_DIR):
        """
        Initialize logger with log directory and file mappings.

        Creates the log directory if it doesn't exist and initializes duplicate prevention cache.
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
        Ensure the log directory exists by creating it if necessary.

        Handles:
        - Creating nested directories
        - Ignoring existing directories
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_caller_info() -> tuple[str, str]:
        """
        Identify the current function and its direct caller in the call stack.

        Walks the call stack to find the first frame outside this logger module.

        Returns:
            tuple: (current_function_name, parent_function_name)
        """
        stack = inspect.stack()
        # Find the first frame not from this logger module
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
        Format a structured log message with contextual information.

        Args:
            level: Log severity level
            message: Primary log message
            error: Exception object (for ERROR logs)
            additional_info: Custom key-value context
            exc_info: Include full exception traceback

        Returns:
            Formatted log string with structured sections
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_function, parent_function = self._get_caller_info()

        # Build log sections
        sections = [
            "=" * 80,
            f"TIMESTAMP: {timestamp}",
            f"LEVEL: {level.value}",
            f"FUNCTION: {current_function}",
            f"PARENT FUNCTION: {parent_function}",
            "-" * 80,
            f"MESSAGE: {message}",
        ]

        # Add error details if provided
        if error:
            sections.extend(
                [
                    f"ERROR TYPE: {type(error).__name__}",
                    f"ERROR MESSAGE: {str(error)}",
                    "-" * 80,
                ]
            )

            # Add traceback if requested
            if exc_info:
                try:
                    tb = getattr(error, "__traceback__", None)
                    trace_lines = (
                        traceback.format_exception(type(error), error, tb)
                        if tb
                        else traceback.format_exc().splitlines()
                    )
                    sections.append("FULL TRACEBACK:")
                    sections.extend(trace_lines)
                    sections.append("-" * 80)
                except Exception as format_err:
                    sections.append(f"Failed to format traceback: {format_err}")

        # Add context information
        context = {"ai_engineer": "Muhammad"}
        if additional_info:
            context.update(additional_info)

        sections.extend(
            [
                "CONTEXT:",
                "\n".join(f"{k}: {v}" for k, v in context.items()),
                "=" * 80 + "\n",
            ]
        )

        return "\n".join(sections)

    def _write_log(self, level: LogLevel, message: str) -> None:
        """
        Write the formatted message to the appropriate log file.

        Prevents duplicate log entries using message hash caching.
        Handles file I/O errors gracefully.
        """
        log_hash = hash(message)
        if log_hash in self._log_cache:
            return  # Skip duplicate logs

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
        Log informational message about system operation.

        Args:
            message: Description of system event
            additional_info: Key-value context about the event
        """
        formatted = self._format_message(
            LogLevel.INFO, message, additional_info=additional_info
        )
        self._write_log(LogLevel.INFO, formatted)

    def warning(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log potential problem that doesn't prevent system operation.

        Args:
            message: Description of warning condition
            additional_info: Context about the warning
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
        Log serious failure that prevents normal operation.

        Args:
            error: Exception that caused the failure
            additional_info: Context about the error state
            exc_info: Include full exception traceback (default: True)
        """
        # Convert exception to a string message
        message = f"{type(error).__name__}: {str(error)}"
        formatted = self._format_message(
            LogLevel.ERROR,
            message,
            error=error,
            additional_info=additional_info,
            exc_info=exc_info,
        )
        self._write_log(LogLevel.ERROR, formatted)

    def exception(
        self,
        message: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log an error message with exception traceback (mimics standard logging.exception).

        Args:
            message: Description of the error
            additional_info: Context about the error state
        """
        _, exc_value, _ = sys.exc_info()
        if exc_value is not None:
            self.error(
                exc_value,
                additional_info=(
                    additional_info if additional_info else {"message": message}
                ),
                exc_info=True,
            )
        else:
            # If called outside an exception context, just log as an error
            self.error(
                Exception(message),
                additional_info=additional_info,
                exc_info=False,
            )

    def debug(
        self, message: str, additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log debugging information (writes to INFO log with DEBUG prefix).

        Args:
            `message`: Debug information
            `additional_info`: Context about debug state
        """
        self.info(f"[DEBUG] {message}", additional_info=additional_info)

    def clear_logs(self, level: Optional[LogLevel] = None) -> None:
        """
        Clear log files while preserving the directory structure.

        Args:
            level: Specific log level to clear (None clears all)
        """
        try:
            targets = [self.log_files[level]] if level else self.log_files.values()
            for file in targets:
                with open(file, "w", encoding="utf-8") as f:
                    f.write("")
            self._log_cache.clear()
        except Exception as e:
            print(f"Failed to clear logs: {e}", file=sys.stderr)


# Global logger instance for system-wide access
system_logger = Logger()


def log_error(message: str, **kwargs) -> None:
    """
    Log the structured error message without exception handling.

    Useful for non-exception error conditions where you want:
    - Structured formatting
    - Context capture
    - But no exception traceback

    Args:
        message: Error description
        **kwargs: Context variables as keyword arguments
    """
    formatted = system_logger._format_message(
        level=LogLevel.ERROR, message=message, additional_info=kwargs, exc_info=False
    )
    system_logger._write_log(LogLevel.ERROR, formatted)
