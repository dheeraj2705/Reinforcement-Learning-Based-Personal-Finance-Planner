"""
app.py

Minimal FastAPI server exposing an endpoint to generate the schedule.

Install dependencies if not present:
    pip install fastapi uvicorn pandas torch

Run:
    uvicorn app:app --reload --port 8000

POST /schedule
JSON body:
{
  "monthly_income": 40000,
  "avg_expense": 12000,
  "plan_months": 6,
  "episodes": 200             # optional, default 200
}

Response:
- JSON containing "schedule" (list of rows) and "csv_path" (if saved)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import pandas as pd

from backend.rl_wrapper import generate_schedule

app = FastAPI(title="Smart Budget Manager - Schedule Planner")

class ScheduleRequest(BaseModel):
    monthly_income: float = Field(..., gt=0)
    avg_expense: float = Field(..., gt=0)
    plan_months: int = Field(..., gt=0)
    episodes: Optional[int] = Field(200, gt=0)
    checkpoint: Optional[str] = Field(None, description="Optional checkpoint path to load/save model")

@app.post("/schedule")
def create_schedule(req: ScheduleRequest):
    try:
        checkpoint = req.checkpoint if req.checkpoint else "checkpoints/scheduler_dqn.pth"
        df = generate_schedule(
            monthly_income=req.monthly_income,
            avg_monthly_expense=req.avg_expense,
            plan_months=req.plan_months,
            checkpoint_path=checkpoint,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save a CSV for the dashboard to pick up (optional)
    os.makedirs("outputs", exist_ok=True)
    csv_path = os.path.join("outputs", "schedule_plan.csv")
    df.to_csv(csv_path, index=False)

    # Return JSON-friendly structure
    return {
        "schedule": df.to_dict(orient="records"),
        "csv_path": csv_path
    }
