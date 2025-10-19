# CaiLL - AI Trading System

**C**ontinuous **AI** **L**earning **L**ab - An advanced AI trading system powered by multi-agent reinforcement learning.

![Status](https://img.shields.io/badge/Status-Phase%205%20Complete-green)
![Frontend](https://img.shields.io/badge/Frontend-Ready-brightgreen)
![Backend](https://img.shields.io/badge/Backend-In%20Progress-yellow)

## 🎯 Overview

CaiLL is a sophisticated algorithmic trading system that uses **reinforcement learning** to train intelligent trading agents. The system features a modern web dashboard for monitoring RL training, portfolio performance, and trading strategies in real-time.

### Key Features

- 🤖 **Multi-Agent RL System** - PPO, A2C, DQN agents working in ensemble
- 📊 **Real-time Dashboard** - Monitor training progress and performance
- 🧠 **Continuous Learning** - Agents improve from live market data
- 📈 **Professional Backtesting** - Institutional-grade strategy validation
- 🎯 **Risk Management** - Multi-layer safety controls
- 🌐 **Modern Stack** - Next.js 15 + Python FastAPI

## 🏗️ Architecture

### Dual-Stack System

```
┌─────────────────────────────────────────┐
│     Next.js Frontend (Port 3000)        │
│   • React 19 + TypeScript               │
│   • Real-time WebSocket (Socket.IO)     │
│   • shadcn/ui components                │
└─────────────────────────────────────────┘
              ↕ WebSocket + REST
┌─────────────────────────────────────────┐
│     Python Backend (Port 8000)          │
│   • FastAPI + RL Engine                 │
│   • Stable-Baselines3 (PPO, A2C, DQN)   │
│   • Gymnasium environments              │
│   • Backtesting & validation            │
└─────────────────────────────────────────┘
```

### Multi-Agent RL Agents

1. **Trend Following Agent** (PPO) - Captures directional market moves
2. **Mean Reversion Agent** (A2C) - Exploits range-bound markets
3. **Volatility Agent** (DQN) - Trades on volatility patterns
4. **Momentum Agent** (PPO) - Short-term momentum strategies

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.9+
- **Docker** (optional, for full stack)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CaiLL.git
cd CaiLL/code

# Install frontend dependencies
npm install

# Install Python dependencies
pip install -r requirements-minimal.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the System

#### Frontend Only (Available Now)
```bash
cd code
npm run dev
```
Open http://localhost:3000

#### Full Stack (Backend + Frontend)
```bash
# Terminal 1: Start Python backend
cd code
python backend/main.py

# Terminal 2: Start frontend
npm run dev
```

#### Docker (Full Stack)
```bash
cd code
docker-compose up -d
```

## 📊 Dashboard Features

### 1. Trading Dashboard
- Real-time portfolio overview
- Active positions tracking
- P&L monitoring
- Agent performance metrics

### 2. RL Training Monitor
- **System-wide metrics**: Total steps, avg rewards, exploration rate
- **Individual agent cards**: Training progress, win rates, Sharpe ratios
- **Experience replay buffer**: Memory utilization monitoring
- **Hyperparameters**: Learning rates, loss metrics, entropy

### 3. Analytics
- Backtesting results
- Performance metrics (30+ indicators)
- Walk-forward analysis
- Monte Carlo simulations

### 4. Settings
- Trading mode configuration
- Risk management parameters
- API key management
- System status

## 🧠 RL Training System

### How Agents Learn

1. **Experience Collection** - Agents trade in simulated environments
2. **Replay Buffer** - Stores state-action-reward transitions
3. **Neural Network Training** - Updates policy and value functions
4. **Policy Improvement** - Tests and refines strategies
5. **Deployment** - Best models deployed for live trading

### Algorithms Used

- **PPO** (Proximal Policy Optimization) - Stable, sample-efficient
- **A2C** (Advantage Actor-Critic) - Fast, on-policy learning
- **DQN** (Deep Q-Network) - Value-based learning

## 📁 Project Structure

```
CaiLL/
├── code/
│   ├── backend/           # Python trading engine
│   │   ├── rl_engine/     # RL agents (PPO, A2C, DQN)
│   │   ├── services/      # Trading, risk, data services
│   │   ├── vector_db/     # Pinecone integration
│   │   └── main.py        # FastAPI server
│   ├── src/
│   │   ├── app/           # Next.js pages
│   │   ├── components/    # React components
│   │   ├── frontend/      # Dashboard views
│   │   ├── backtesting/   # Strategy validation
│   │   └── lib/           # Utilities
│   ├── prisma/            # Database schema
│   └── docker-compose.yml # Container orchestration
├── CLAUDE.md              # AI assistant documentation
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `code/` directory:

```env
# Database
DATABASE_URL="file:./dev.db"

# Trading
ENABLE_TRADING=false
PAPER_TRADING=true

# API Keys
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
PINECONE_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

## 🧪 Testing

### Frontend
```bash
npm run lint
npm run build
```

### Backend
```bash
pytest backend/rl_engine/test_rl_engine.py -v
python test_basic.py  # Environment validation
```

## 📈 Current Status

### ✅ Completed (Phase 5)
- [x] Next.js frontend with full dashboard
- [x] RL training monitoring interface
- [x] Trading view with portfolio tracking
- [x] Backtesting engine with validation
- [x] Monte Carlo simulation
- [x] Walk-forward analysis
- [x] Performance calculator (30+ metrics)
- [x] Socket.IO real-time communication

### 🚧 In Progress
- [ ] Fix Python backend module imports
- [ ] Backend-frontend integration
- [ ] Live data streaming
- [ ] Real-time RL training

### 📋 Planned (Phase 6+)
- [ ] Educational modules
- [ ] Advanced visualizations
- [ ] Model explainability
- [ ] Production deployment

## 🔒 Safety Features

- **Paper Trading Mode** - Test with virtual money first
- **Risk Limits** - Maximum drawdown, position size controls
- **Circuit Breakers** - Automatic shutdown on unusual conditions
- **Multi-layer Validation** - Statistical significance testing
- **Audit Trail** - Complete logging of all decisions

## 🤝 Contributing

This is a learning and research project. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always start with paper trading
- Never risk money you cannot afford to lose
- This is NOT financial advice
- Consult professional financial advisors

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete architecture guide
- **[QUICK_TEST.md](code/QUICK_TEST.md)** - Quick start testing
- **[TESTING_SUMMARY.md](code/TESTING_SUMMARY.md)** - Test results
- **[RL_TRAINING_DASHBOARD.md](code/RL_TRAINING_DASHBOARD.md)** - RL monitoring guide
- **[UI_READY.md](code/UI_READY.md)** - UI feature documentation

## 🛠️ Technology Stack

### Frontend
- Next.js 15 (React 19)
- TypeScript 5
- Tailwind CSS 4
- shadcn/ui
- Socket.IO
- TanStack Query
- Zustand

### Backend
- Python 3.9+
- FastAPI
- Stable-Baselines3
- Gymnasium
- PyTorch
- Pandas, NumPy
- InfluxDB, PostgreSQL, Redis
- Pinecone (vector DB)

### Infrastructure
- Docker & Docker Compose
- Prometheus & Grafana
- RabbitMQ
- MLflow

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Stable-Baselines3 team for RL algorithms
- OpenAI Gymnasium for RL environments
- shadcn for UI components
- Next.js and FastAPI teams

---

**Built with ❤️ for algorithmic trading research and education**

For questions or support, please open an issue on GitHub.
