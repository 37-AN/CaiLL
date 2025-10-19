# Quick Test & Run Guide

## Current Status

✅ **Frontend is RUNNING** on http://localhost:3000

✅ **Dashboard UI is READY** - Full trading interface with tabs

⚠️ **Backend needs import fixes** before it can run

## What's Working Right Now

### Access the Frontend
The Next.js frontend is already running! Just open your browser:

```
http://localhost:3000
```

or

```
http://127.0.0.1:3000
```

**Features Available:**
- ✅ **Trading Tab** - Portfolio overview, positions, agent status
- ✅ **Learning Tab** - Educational modules and tutorials
- ✅ **Analytics Tab** - Backtesting and performance metrics
- ✅ **Settings Tab** - System configuration
- Modern UI with shadcn/ui components
- Real-time Socket.IO WebSocket connection
- Responsive design with Tailwind CSS
- Dark/light mode support

### Test Python Environment
```bash
cd code
python test_basic.py
```

**Expected Output:**
```
[OK] NumPy: 2.2.6
[OK] Pandas: 2.3.2
[OK] Gymnasium: 1.2.1
[OK] Stable-Baselines3: 2.7.0
[OK] PyTorch: 2.8.0+cpu
[OK] FastAPI
[OK] Created Gymnasium environment
```

## What's NOT Working (Yet)

### Backend Server
The Python backend has import path issues. To fix:

1. **Option A - Run without full imports:**
   Create a minimal FastAPI server that doesn't import the RL modules yet

2. **Option B - Fix the imports:**
   Add proper `PYTHONPATH` or restructure the backend package imports

## Quick Commands

### Frontend
```bash
# Already running! Check status:
curl http://localhost:3000

# If you need to restart:
# (Stop with Ctrl+C in the terminal where it's running)
cd code
npm run dev
```

### Database
```bash
# Reset database
cd code
npm run db:reset

# Generate Prisma client
npm run db:generate

# Push schema changes
npm run db:push
```

### Docker (Full Stack)
```bash
cd code
docker-compose up -d      # Start all services
docker-compose logs -f    # View logs
docker-compose down       # Stop all services
```

## What You Can Do Right Now

1. **Explore the Frontend**
   - Open http://localhost:3000
   - Check browser DevTools for Socket.IO connection
   - Inspect the React components

2. **Modify UI Components**
   - Edit files in `code/src/components/ui/`
   - Changes will hot-reload automatically

3. **Test Python Libraries**
   - Run `python test_basic.py`
   - Import and test ML libraries in Python REPL

4. **Read Documentation**
   - See `CLAUDE.md` for architecture details
   - See `TESTING_SUMMARY.md` for full test results
   - See `README.md` for project overview

## Next Steps to Get Backend Running

1. Fix Python imports in `backend/rl_engine/`
2. Start backend: `python backend/main.py`
3. Access API docs: http://localhost:8000/docs
4. Test integration between frontend and backend

## Need Help?

**Frontend Issues:**
- Check `code/dev.log` for errors
- Verify node_modules exists
- Run `npm install` if needed

**Backend Issues:**
- Verify Python packages: `pip list`
- Install minimal deps: `pip install -r requirements-minimal.txt`
- Check imports: `python -c "import gymnasium; print('OK')"`

**Docker Issues:**
- Check Docker is running: `docker --version`
- View container logs: `docker-compose logs [service-name]`
- Restart services: `docker-compose restart`

## Files Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Complete architecture guide |
| `TESTING_SUMMARY.md` | Detailed test results |
| `QUICK_TEST.md` | This file - quick reference |
| `test_basic.py` | Python environment test |
| `.env` | Environment variables |
| `dev.log` | Frontend development log |

## Environment Variables

Located in `code/.env`:
- `DATABASE_URL` - SQLite database path
- `NEXTAUTH_SECRET` - Auth secret
- API keys for trading services (add your own)

## Ports in Use

- **3000** - Next.js frontend (ACTIVE)
- **8000** - Python backend (not started)
- **5432** - PostgreSQL (Docker)
- **6379** - Redis (Docker)
- **8086** - InfluxDB (Docker)

---

**Remember**: The frontend is running NOW at http://localhost:3000 - go check it out!
