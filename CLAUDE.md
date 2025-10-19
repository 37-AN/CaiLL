# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**CaiLL (AI Learning Lab)** is a sophisticated AI Trading System that combines reinforcement learning with modern web technologies. The project has two major components:

1. **Python Backend** - FastAPI-based trading system with RL agents, backtesting, and risk management
2. **Next.js Frontend** - Modern TypeScript/React dashboard for monitoring, control, and education

## Architecture

### Dual-Stack System

```
┌─────────────────────────────────────────────────────────────┐
│                   Next.js Frontend (Port 3000)              │
│   • React 19 + TypeScript + Tailwind                        │
│   • shadcn/ui components + Socket.IO client                 │
│   • Educational modules & interactive dashboards            │
└─────────────────────────────────────────────────────────────┘
                            ↕ (WebSocket + REST)
┌─────────────────────────────────────────────────────────────┐
│                Python Backend (Port 8000)                   │
│   • FastAPI + Reinforcement Learning Engine                 │
│   • Multi-agent RL system (Stable-Baselines3)               │
│   • Vector DB (Pinecone), Time Series (InfluxDB)            │
│   • Backtesting + Monte Carlo simulation                    │
│   • Risk Management + Position sizing                       │
└─────────────────────────────────────────────────────────────┘
```

### Backend Structure

```
backend/
├── core/              # Configuration, logging, exceptions
├── rl_engine/         # Reinforcement learning agents
│   ├── trading_env.py          # Gymnasium trading environment
│   ├── multi_agent_rl.py       # Multi-agent coordination
│   ├── continuous_learning.py  # Online learning & adaptation
│   └── reward_functions.py     # Reward shaping
├── services/          # Core services (market data, trading engine, risk)
├── vector_db/         # Pinecone integration for pattern recognition
├── data_collectors/   # Market data ingestion (Alpaca, Binance, etc.)
└── features/          # Feature engineering for ML

src/
├── backtesting/       # Backtesting & validation system
│   ├── backtest_engine.py      # Event-driven backtesting
│   ├── walk_forward.py         # Walk-forward analysis
│   ├── monte_carlo.py          # Monte Carlo simulation
│   ├── strategy_validation.py  # Statistical validation
│   ├── performance_calculator.py # 30+ metrics
│   └── pipeline.py             # Complete analysis pipeline
├── education/         # Educational modules & explainability
├── execution/         # Order execution & broker integration
└── risk_management/   # Risk controls & position sizing
```

### Frontend Structure

```
src/
├── app/               # Next.js 15 App Router
│   ├── api/          # API routes & Socket.IO endpoint
│   └── page.tsx      # Main dashboard
├── components/        # React components
│   └── ui/           # shadcn/ui components
├── hooks/            # Custom React hooks
└── lib/              # Utilities & Socket.IO setup
```

## Development Commands

### Frontend (Next.js)

```bash
# Development with hot reload & logging
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Lint code
npm run lint

# Database operations (Prisma)
npm run db:push      # Push schema to DB
npm run db:generate  # Generate Prisma client
npm run db:migrate   # Run migrations
npm run db:reset     # Reset database
```

### Backend (Python)

```bash
# Start FastAPI backend
python backend/main.py

# Or with uvicorn directly
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest backend/rl_engine/test_rl_engine.py
pytest -v --cov=backend tests/

# Code quality
black backend/
isort backend/
flake8 backend/
mypy backend/
```

### Docker Compose (Full Stack)

```bash
# Start all services (PostgreSQL, Redis, InfluxDB, RabbitMQ, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop all services
docker-compose down

# Rebuild containers
docker-compose up -d --build
```

## Key Technologies

### Backend
- **FastAPI** - Async Python web framework
- **Stable-Baselines3** - Reinforcement learning library (PPO, A2C, DQN)
- **Gymnasium** - RL environment interface (OpenAI Gym successor)
- **Ray RLlib** - Distributed RL training
- **Pinecone** - Vector database for pattern matching
- **InfluxDB** - Time-series database for market data
- **PostgreSQL** - Relational data (trades, positions)
- **Redis** - Caching & pub/sub
- **RabbitMQ** - Message queue for async tasks
- **Trading APIs**: Alpaca (stocks), Binance (crypto), Interactive Brokers

### Frontend
- **Next.js 15** - React framework with App Router
- **React 19** - Latest React with Server Components
- **TypeScript 5** - Type-safe JavaScript
- **Tailwind CSS 4** - Utility-first styling
- **shadcn/ui** - Radix UI components
- **Socket.IO** - Real-time bidirectional communication
- **TanStack Query** - Data fetching & caching
- **Zustand** - State management
- **Framer Motion** - Animations

## Critical Architecture Patterns

### 1. Multi-Agent RL System

The backend uses **multiple specialized RL agents** that work together:

- **Trend Following Agent** - Identifies and follows market trends
- **Mean Reversion Agent** - Exploits price reversion to mean
- **Volatility Agent** - Trades on volatility patterns
- **Momentum Agent** - Captures short-term momentum

Each agent is trained independently but coordinated by an orchestrator that:
- Aggregates agent predictions
- Manages agent weights based on recent performance
- Prevents conflicting signals

**Key Files**: `backend/rl_engine/multi_agent_rl.py`, `backend/rl_engine/continuous_learning.py`

### 2. Continuous Learning Pipeline

The system **learns continuously** from live market data:

1. **Experience Replay Buffer** - Stores recent market interactions
2. **Online Training** - Periodic model updates with new data
3. **Performance Monitoring** - Detects degradation and triggers retraining
4. **Version Control** - MLflow integration for model versioning

**Key Files**: `backend/rl_engine/continuous_learning.py`, `backend/services/trading_engine.py`

### 3. Backtesting System

Production-grade backtesting with **institutional-level validation**:

- **Event-Driven Simulation** - Realistic tick-by-tick execution
- **Walk-Forward Analysis** - Temporal robustness testing
- **Monte Carlo Simulation** - 1000+ scenarios for uncertainty quantification
- **Statistical Validation** - Hypothesis testing with multiple testing corrections
- **30+ Performance Metrics** - Sharpe, Sortino, Calmar, VaR, CVaR, etc.

**Key Files**: All files in `src/backtesting/`

### 4. Real-Time Communication

Custom Next.js server integrates **Socket.IO** for bidirectional communication:

- **Server**: `server.ts` creates HTTP server with both Next.js and Socket.IO
- **Client**: `src/lib/socket.ts` manages WebSocket connections
- **Use Cases**: Real-time trade updates, portfolio monitoring, agent status

### 5. Vector Database for Pattern Recognition

**Pinecone** stores embeddings of historical market patterns:

1. Market data → Feature extraction → Sentence transformers
2. Store embeddings in Pinecone with metadata
3. Query similar patterns during live trading
4. Use pattern history to inform RL agent decisions

**Key Files**: `backend/vector_db/`, `backend/services/pinecone_client.py`

## Important Development Notes

### Environment Setup

1. **Python Environment**: Requires Python 3.9+ (3.10 recommended)
2. **Node.js**: Requires Node.js 18+ for Next.js 15
3. **Environment Variables**: Create `.env` file with:
   ```
   # Database
   DATABASE_URL="postgresql://postgres:password@localhost:5432/trading_system"

   # APIs
   PINECONE_API_KEY=your_key
   ALPHA_VANTAGE_API_KEY=your_key
   ALPACA_API_KEY=your_key
   ALPACA_SECRET_KEY=your_secret

   # Backend Config
   ENABLE_TRADING=false  # Start with paper trading
   PAPER_TRADING=true
   DEBUG=true
   ```

### Running the Full System

**Development Mode** (recommended for coding):

```bash
# Terminal 1: Start backend
cd code
python backend/main.py

# Terminal 2: Start frontend
npm run dev

# Terminal 3: Start supporting services (optional)
docker-compose up redis influxdb postgres
```

**Production Mode** (Docker):

```bash
docker-compose up -d
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- Grafana: http://localhost:3000 (Grafana port conflicts with Next.js - change if needed)

### Testing Strategies

**Backend Testing**:
```bash
# Test RL engine
pytest backend/rl_engine/test_rl_engine.py -v

# Test backtesting
pytest tests/backtesting/ -v

# Integration tests
pytest tests/integration/ -v
```

**Frontend Testing**:
- Component tests would typically use Jest + React Testing Library (not currently set up)
- Use browser DevTools for Socket.IO debugging

### Database Migrations

**Prisma** (Frontend - SQLite by default):
```bash
npm run db:migrate   # Create new migration
npm run db:push      # Push without migration (dev only)
```

**Alembic** (Backend - PostgreSQL):
```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

## Safety & Risk Management

### Critical Safety Rules

1. **ALWAYS start with paper trading** (`PAPER_TRADING=true`)
2. **NEVER commit API keys** - Use `.env` files (already in `.gitignore`)
3. **Test strategies thoroughly** - Minimum 30 days paper trading before live
4. **Respect risk limits** - Set in `backend/services/risk_manager.py`
5. **Monitor continuously** - Use Grafana dashboards for alerts

### Built-in Safety Features

- **Maximum Drawdown Limits** - Circuit breaker if losses exceed threshold
- **Position Size Limits** - Prevent over-concentration
- **Portfolio-wide Risk Limits** - Total exposure caps
- **Emergency Stop** - Immediate shutdown via API endpoint
- **Slippage & Commission Modeling** - Realistic execution costs in backtesting

## Common Tasks

### Adding a New RL Agent

1. Create agent class in `backend/rl_engine/multi_agent_rl.py`
2. Define observation/action space in `action_spaces.py`
3. Implement custom reward function in `reward_functions.py`
4. Register agent in `TradingEngine` service
5. Add agent to orchestrator coordination logic

### Adding a New Performance Metric

1. Add calculation to `src/backtesting/performance_calculator.py`
2. Update `calculate_all_metrics()` method
3. Add visualization to `visualization.py`
4. Include in pipeline output

### Creating New Educational Module

1. Add module content in `src/education/modules/`
2. Implement interactive components in frontend
3. Add Socket.IO events for real-time interaction
4. Update `backend/services/education_engine.py`

## External Documentation

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Next.js 15**: https://nextjs.org/docs
- **shadcn/ui**: https://ui.shadcn.com/
- **Pinecone**: https://docs.pinecone.io/
- **Alpaca API**: https://alpaca.markets/docs/

## Project Context

This project is designed as an **educational AI trading system** per `files/QUICK_START_GUIDE.md`. The system demonstrates:

- Reinforcement learning for trading
- Backtesting methodology
- Risk management principles
- Software engineering for financial systems

**Current Status**: Phase 5 Complete (Backtesting & Validation)
**Next Phase**: Phase 6 - Educational System & User Interface

Refer to `code/PHASE_5_COMPLETE.md` for detailed completion status.
