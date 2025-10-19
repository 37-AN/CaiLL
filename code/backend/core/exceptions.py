"""
Custom exceptions for the AI Trading System.

This module defines all custom exceptions used throughout the application
to provide better error handling and user feedback.
"""

from typing import Any, Dict, Optional


class TradingSystemException(Exception):
    """
    Base exception for all trading system errors.
    """
    
    def __init__(
        self,
        message: str,
        error_type: str = "trading_system_error",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_type = error_type
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(TradingSystemException):
    """
    Raised when there's a configuration error.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="configuration_error",
            status_code=500,
            details=details
        )


class DatabaseError(TradingSystemException):
    """
    Raised when there's a database-related error.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="database_error",
            status_code=500,
            details=details
        )


class DataCollectionError(TradingSystemException):
    """
    Raised when there's an error collecting market data.
    """
    
    def __init__(self, message: str, data_source: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["data_source"] = data_source
        super().__init__(
            message=message,
            error_type="data_collection_error",
            status_code=503,
            details=details
        )


class DataValidationError(TradingSystemException):
    """
    Raised when market data fails validation.
    """
    
    def __init__(self, message: str, symbol: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["symbol"] = symbol
        super().__init__(
            message=message,
            error_type="data_validation_error",
            status_code=400,
            details=details
        )


class TradingError(TradingSystemException):
    """
    Base exception for trading-related errors.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="trading_error",
            status_code=400,
            details=details
        )


class OrderExecutionError(TradingError):
    """
    Raised when order execution fails.
    """
    
    def __init__(self, message: str, symbol: str, order_type: str, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "symbol": symbol,
            "order_type": order_type
        })
        super().__init__(
            message=message,
            error_type="order_execution_error",
            status_code=400,
            details=details
        )


class InsufficientFundsError(TradingError):
    """
    Raised when there are insufficient funds for a trade.
    """
    
    def __init__(self, message: str, required_amount: float, available_amount: float,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "required_amount": required_amount,
            "available_amount": available_amount
        })
        super().__init__(
            message=message,
            error_type="insufficient_funds_error",
            status_code=400,
            details=details
        )


class PositionNotFoundError(TradingError):
    """
    Raised when a requested position is not found.
    """
    
    def __init__(self, message: str, symbol: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["symbol"] = symbol
        super().__init__(
            message=message,
            error_type="position_not_found_error",
            status_code=404,
            details=details
        )


class RiskManagementError(TradingSystemException):
    """
    Raised when a risk management rule is violated.
    """
    
    def __init__(self, message: str, risk_type: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["risk_type"] = risk_type
        super().__init__(
            message=message,
            error_type="risk_management_error",
            status_code=400,
            details=details
        )


class MaxDrawdownExceededError(RiskManagementError):
    """
    Raised when maximum drawdown is exceeded.
    """
    
    def __init__(self, message: str, current_drawdown: float, max_drawdown: float,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown
        })
        super().__init__(
            message=message,
            risk_type="max_drawdown_exceeded",
            status_code=400,
            details=details
        )


class PositionSizeExceededError(RiskManagementError):
    """
    Raised when position size exceeds limits.
    """
    
    def __init__(self, message: str, requested_size: float, max_size: float,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "requested_size": requested_size,
            "max_size": max_size
        })
        super().__init__(
            message=message,
            risk_type="position_size_exceeded",
            status_code=400,
            details=details
        )


class PortfolioRiskExceededError(RiskManagementError):
    """
    Raised when portfolio risk exceeds limits.
    """
    
    def __init__(self, message: str, current_risk: float, max_risk: float,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "current_risk": current_risk,
            "max_risk": max_risk
        })
        super().__init__(
            message=message,
            risk_type="portfolio_risk_exceeded",
            status_code=400,
            details=details
        )


class ModelError(TradingSystemException):
    """
    Base exception for ML model errors.
    """
    
    def __init__(self, message: str, model_name: str, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["model_name"] = model_name
        super().__init__(
            message=message,
            error_type="model_error",
            status_code=500,
            details=details
        )


class ModelTrainingError(ModelError):
    """
    Raised when model training fails.
    """
    
    def __init__(self, message: str, model_name: str, epoch: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if epoch is not None:
            details["epoch"] = epoch
        super().__init__(
            message=message,
            model_name=model_name,
            error_type="model_training_error",
            status_code=500,
            details=details
        )


class ModelPredictionError(ModelError):
    """
    Raised when model prediction fails.
    """
    
    def __init__(self, message: str, model_name: str, input_data: Optional[Dict[str, Any]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if input_data is not None:
            details["input_data"] = input_data
        super().__init__(
            message=message,
            model_name=model_name,
            error_type="model_prediction_error",
            status_code=500,
            details=details
        )


class AuthenticationError(TradingSystemException):
    """
    Raised when authentication fails.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_type="authentication_error",
            status_code=401,
            details=details
        )


class AuthorizationError(TradingSystemException):
    """
    Raised when authorization fails.
    """
    
    def __init__(self, message: str, required_permission: str, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["required_permission"] = required_permission
        super().__init__(
            message=message,
            error_type="authorization_error",
            status_code=403,
            details=details
        )


class APIError(TradingSystemException):
    """
    Raised when an external API call fails.
    """
    
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["api_name"] = api_name
        if status_code is not None:
            details["api_status_code"] = status_code
        super().__init__(
            message=message,
            error_type="api_error",
            status_code=502,
            details=details
        )


class RateLimitError(APIError):
    """
    Raised when API rate limit is exceeded.
    """
    
    def __init__(self, message: str, api_name: str, retry_after: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            api_name=api_name,
            status_code=429,
            error_type="rate_limit_error",
            details=details
        )


class ValidationError(TradingSystemException):
    """
    Raised when input validation fails.
    """
    
    def __init__(self, message: str, field: str, value: Any,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "field": field,
            "value": value
        })
        super().__init__(
            message=message,
            error_type="validation_error",
            status_code=422,
            details=details
        )


class NotFoundError(TradingSystemException):
    """
    Raised when a requested resource is not found.
    """
    
    def __init__(self, message: str, resource_type: str, resource_id: str,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })
        super().__init__(
            message=message,
            error_type="not_found_error",
            status_code=404,
            details=details
        )


class CircuitBreakerError(TradingSystemException):
    """
    Raised when circuit breaker is open.
    """
    
    def __init__(self, message: str, service_name: str, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["service_name"] = service_name
        super().__init__(
            message=message,
            error_type="circuit_breaker_error",
            status_code=503,
            details=details
        )


class BacktestingError(TradingSystemException):
    """
    Raised when backtesting fails.
    """
    
    def __init__(self, message: str, strategy_name: str, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["strategy_name"] = strategy_name
        super().__init__(
            message=message,
            error_type="backtesting_error",
            status_code=500,
            details=details
        )


class EducationalContentError(TradingSystemException):
    """
    Raised when educational content generation fails.
    """
    
    def __init__(self, message: str, content_type: str, 
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["content_type"] = content_type
        super().__init__(
            message=message,
            error_type="educational_content_error",
            status_code=500,
            details=details
        )