import logging
import sys

import structlog


def configure_logging() -> None:
    """Configure structured logging for the application."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound structured logger."""
    return structlog.get_logger(name)
