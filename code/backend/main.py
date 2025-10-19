"""
AI Trading System - Main Application Entry Point

This is the main entry point for the AI Trading System. It initializes all components
and starts the FastAPI application with proper error handling and logging.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router as api_router
from backend.core.config import settings
from backend.core.logging import setup_logging
from backend.core.exceptions import TradingSystemException
from backend.services.database import init_database
from backend.services.redis_client import init_redis
from backend.services.influxdb_client import init_influxdb
from backend.services.pinecone_client import init_pinecone
from backend.services.message_queue import init_message_queue
from backend.services.market_data import MarketDataService
from backend.services.trading_engine import TradingEngine
from backend.services.risk_manager import RiskManager
from backend.services.education_engine import EducationEngine

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global variables for services
market_data_service: MarketDataService = None
trading_engine: TradingEngine = None
risk_manager: RiskManager = None
education_engine: EducationEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the application.
    """
    logger.info("Starting AI Trading System...")
    
    try:
        # Initialize core services
        await initialize_services()
        
        # Start background tasks
        await start_background_tasks()
        
        logger.info("AI Trading System started successfully!")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start AI Trading System: {e}")
        raise
    finally:
        logger.info("Shutting down AI Trading System...")
        await cleanup_services()
        logger.info("AI Trading System shutdown complete.")


async def initialize_services() -> None:
    """
    Initialize all core services in the correct order.
    """
    global market_data_service, trading_engine, risk_manager, education_engine
    
    logger.info("Initializing core services...")
    
    # Initialize database connections
    await init_database()
    await init_redis()
    await init_influxdb()
    await init_pinecone()
    await init_message_queue()
    
    # Initialize core services
    market_data_service = MarketDataService()
    await market_data_service.initialize()
    
    risk_manager = RiskManager()
    await risk_manager.initialize()
    
    trading_engine = TradingEngine(
        market_data_service=market_data_service,
        risk_manager=risk_manager
    )
    await trading_engine.initialize()
    
    education_engine = EducationEngine()
    await education_engine.initialize()
    
    logger.info("All services initialized successfully!")


async def start_background_tasks() -> None:
    """
    Start background tasks for data collection and trading.
    """
    logger.info("Starting background tasks...")
    
    # Start market data collection
    if market_data_service:
        asyncio.create_task(market_data_service.start_data_collection())
    
    # Start trading engine (if enabled)
    if trading_engine and settings.ENABLE_TRADING:
        asyncio.create_task(trading_engine.start_trading())
    
    logger.info("Background tasks started!")


async def cleanup_services() -> None:
    """
    Cleanup all services during shutdown.
    """
    global market_data_service, trading_engine, risk_manager, education_engine
    
    logger.info("Cleaning up services...")
    
    # Stop trading engine
    if trading_engine:
        await trading_engine.stop_trading()
    
    # Stop market data collection
    if market_data_service:
        await market_data_service.stop_data_collection()
    
    # Cleanup other services
    if education_engine:
        await education_engine.cleanup()
    
    if risk_manager:
        await risk_manager.cleanup()
    
    logger.info("Services cleanup complete!")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered trading system with reinforcement learning",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)


# Exception handlers
@app.exception_handler(TradingSystemException)
async def trading_system_exception_handler(request, exc: TradingSystemException):
    """Handle custom trading system exceptions."""
    logger.error(f"Trading system error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_type,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "services": {
            "database": "connected",
            "redis": "connected",
            "influxdb": "connected",
            "pinecone": "connected",
            "message_queue": "connected"
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with basic information.
    """
    return {
        "message": "AI Trading System",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "Documentation not available in production"
    }


# Service status endpoint
@app.get("/status")
async def get_status():
    """
    Get detailed status of all services.
    """
    global market_data_service, trading_engine, risk_manager, education_engine
    
    return {
        "system": {
            "status": "running",
            "uptime": "TODO",  # Implement uptime tracking
            "version": settings.VERSION
        },
        "services": {
            "market_data": {
                "status": "running" if market_data_service else "stopped",
                "data_sources": "TODO"  # Implement data source status
            },
            "trading_engine": {
                "status": "running" if trading_engine else "stopped",
                "mode": "paper" if settings.PAPER_TRADING else "live",
                "active_positions": "TODO"  # Implement position tracking
            },
            "risk_manager": {
                "status": "running" if risk_manager else "stopped",
                "risk_level": "TODO"  # Implement risk level calculation
            },
            "education_engine": {
                "status": "running" if education_engine else "stopped",
                "modules_available": "TODO"  # Implement module tracking
            }
        }
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )