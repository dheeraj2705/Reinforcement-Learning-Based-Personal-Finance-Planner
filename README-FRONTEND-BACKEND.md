Smart Budget Manager — RL-based Goal & Schedule Planner

Prereqs
- Python 3.10+ (3.13 ok)
- Node.js 18+

Backend (FastAPI)
1) Create/activate venv (optional)
2) Install deps:
   pip install -r backend/requirements-backend.txt -r requirements.txt
3) Run server:
   uvicorn backend.main:app --reload
4) Open API docs: http://localhost:8000/docs

Demo mode (FAST_MODE)
- For quick UI testing (no RL load/train), enable FAST_MODE:
  set FAST_MODE=true  (Windows PowerShell: $env:FAST_MODE="true")
  This uses a heuristic schedule generator.

Core endpoints
- GET /health
- POST /api/goal/generate
- POST /api/schedule/generate
- POST /api/plan/save
- GET  /api/plan/list
- GET  /api/plan/{plan_id}
- POST /api/plan/{plan_id}/progress
- POST /api/plan/{plan_id}/restructure
- POST /api/upload/csv
- GET  /api/download/plan/{plan_id}

Notes
- The backend generates schedules via DQN (see `backend/rl_wrapper.py`) and loads checkpoint from `checkpoints/scheduler_dqn.pth`. If missing, it falls back to a heuristic.
- Saved CSVs and metadata live in outputs/.

Frontend (React + Vite + Tailwind)
1) cd frontend
2) npm install
3) npm run dev
4) Open http://localhost:5173

Pages
- Landing: gradient hero with feature cards and CTAs.
- Planner: tabs for Goal Planner and Schedule Planner; generate, preview, download, and save.
- Track: select saved plan, upload actuals CSV or inline edit, charts update, re-plan.

Configure API URL
- By default, frontend expects backend at http://localhost:8000
  Set VITE_API_URL in frontend/.env if different.

Example requests

Generate schedule
curl -X POST http://localhost:8000/api/schedule/generate \
  -H "Content-Type: application/json" \
  -d '{"monthly_income":40000,"avg_monthly_expense":20000,"plan_months":12,"title":"Emergency Fund"}'

Save plan
curl -X POST http://localhost:8000/api/plan/save \
  -H "Content-Type: application/json" \
  -d '{"title":"Emergency Fund","notes":"demo","plan":[{"Month":1,"Period":1,"Income":40000,"ExpenseNeed":20000,"SpendAmt":18000,"SaveAmt":16000,"InvestAmt":6000,"Wealth":16000}],"monthly_income":40000,"avg_monthly_expense":20000,"plan_months":12}'

Restructure
curl -X POST http://localhost:8000/api/plan/PLAN_ID/restructure \
  -H "Content-Type: application/json" \
  -d '{"actuals":[{"month":1,"actual_income":40000,"actual_spend":21000,"actual_save":16000,"actual_invest":3000}]}'


