"""
Logging configuration for the AI Trading System.

This module sets up structured logging with proper formatting,
log levels, and output destinations.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from backend.core.config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    # Use settings if not provided
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    if log_file is None:
        log_file = "logs/trading_system.log"
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        settings.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler for errors and above
    if enable_file:
        error_file_handler = logging.handlers.RotatingFileHandler(
            log_path.parent / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(formatter)
        root_logger.addHandler(error_file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """
    Specialized logger for trading operations.
    """
    
    def __init__(self, name: str = "trading"):
        self.logger = get_logger(name)
    
    def trade_executed(self, symbol: str, action: str, quantity: int, price: float, 
                      strategy: str, confidence: float) -> None:
        """Log trade execution."""
        self.logger.info(
            f"TRADE_EXECUTED: {action} {quantity} {symbol} @ ${price:.2f} "
            f"Strategy: {strategy}, Confidence: {confidence:.2f}"
        )
    
    def signal_generated(self, symbol: str, signal: str, strength: float, 
                        indicators: dict) -> None:
        """Log signal generation."""
        self.logger.info(
            f"SIGNAL_GENERATED: {signal} for {symbol}, Strength: {strength:.2f}, "
            f"Indicators: {indicators}"
        )
    
    def risk_alert(self, alert_type: str, message: str, metrics: dict) -> None:
        """Log risk alerts."""
        self.logger.warning(
            f"RISK_ALERT: {alert_type} - {message}, Metrics: {metrics}"
        )
    
    def portfolio_update(self, total_value: float, cash: float, 
                        positions: dict, pnl: float) -> None:
        """Log portfolio updates."""
        self.logger.info(
            f"PORTFOLIO_UPDATE: Total: ${total_value:.2f}, Cash: ${cash:.2f}, "
            f"P&L: ${pnl:.2f}, Positions: {len(positions)}"
        )
    
    def model_performance(self, model_name: str, accuracy: float, 
                         sharpe_ratio: float, max_drawdown: float) -> None:
        """Log model performance metrics."""
        self.logger.info(
            f"MODEL_PERFORMANCE: {model_name} - Accuracy: {accuracy:.3f}, "
            f"Sharpe: {sharpe_ratio:.3f}, Max DD: {max_drawdown:.3f}"
        )
    
    def data_quality(self, data_source: str, quality_score: float, 
                    issues: list) -> None:
        """Log data quality metrics."""
        if quality_score < 0.9:
            self.logger.warning(
                f"DATA_QUALITY: {data_source} - Score: {quality_score:.3f}, "
                f"Issues: {issues}"
            )
        else:
            self.logger.info(
                f"DATA_QUALITY: {data_source} - Score: {quality_score:.3f}"
            )
    
    def system_event(self, event_type: str, message: str, 
                    component: str) -> None:
        """Log system events."""
        self.logger.info(
            f"SYSTEM_EVENT: {event_type} in {component} - {message}"
        )
    
    def error(self, error_type: str, message: str, 
              component: str, exception: Optional[Exception] = None) -> None:
        """Log errors with optional exception details."""
        if exception:
            self.logger.error(
                f"ERROR: {error_type} in {component} - {message}. "
                f"Exception: {str(exception)}",
                exc_info=True
            )
        else:
            self.logger.error(
                f"ERROR: {error_type} in {component} - {message}"
            )


class PerformanceLogger:
    """
    Specialized logger for performance monitoring.
    """
    
    def __init__(self, name: str = "performance"):
        self.logger = get_logger(name)
    
    def api_request(self, endpoint: str, method: str, 
                   response_time: float, status_code: int) -> None:
        """Log API request performance."""
        self.logger.info(
            f"API_REQUEST: {method} {endpoint} - {response_time:.3f}s "
            f"Status: {status_code}"
        )
    
    def database_query(self, query_type: str, execution_time: float, 
                      rows_affected: int) -> None:
        """Log database query performance."""
        self.logger.info(
            f"DB_QUERY: {query_type} - {execution_time:.3f}s, "
            f"Rows: {rows_affected}"
        )
    
    def ml_prediction(self, model_name: str, prediction_time: float, 
                     confidence: float) -> None:
        """Log ML prediction performance."""
        self.logger.info(
            f"ML_PREDICTION: {model_name} - {prediction_time:.3f}s, "
            f"Confidence: {confidence:.3f}"
        )
    
    def data_processing(self, operation: str, processing_time: float, 
                       records_processed: int) -> None:
        """Log data processing performance."""
        self.logger.info(
            f"DATA_PROCESSING: {operation} - {processing_time:.3f}s, "
            f"Records: {records_processed}"
        )


# Create global logger instances
trading_logger = TradingLogger()
performance_logger = PerformanceLogger()


def log_trade_execution(symbol: str, action: str, quantity: int, price: float,
                       strategy: str, confidence: float) -> None:
    """Convenience function for logging trade execution."""
    trading_logger.trade_executed(symbol, action, quantity, price, strategy, confidence)


def log_signal_generation(symbol: str, signal: str, strength: float,
                         indicators: dict) -> None:
    """Convenience function for logging signal generation."""
    trading_logger.signal_generated(symbol, signal, strength, indicators)


def log_risk_alert(alert_type: str, message: str, metrics: dict) -> None:
    """Convenience function for logging risk alerts."""
    trading_logger.risk_alert(alert_type, message, metrics)


def log_portfolio_update(total_value: float, cash: float, positions: dict, pnl: float) -> None:
    """Convenience function for logging portfolio updates."""
    trading_logger.portfolio_update(total_value, cash, positions, pnl)


def log_model_performance(model_name: str, accuracy: float, sharpe_ratio: float,
                         max_drawdown: float) -> None:
    """Convenience function for logging model performance."""
    trading_logger.model_performance(model_name, accuracy, sharpe_ratio, max_drawdown)


def log_system_error(error_type: str, message: str, component: str,
                    exception: Optional[Exception] = None) -> None:
    """Convenience function for logging system errors."""
    trading_logger.error(error_type, message, component, exception)