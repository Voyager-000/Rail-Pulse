"""
RailPulse - FastAPI Backend
Serves risk predictions as REST endpoints for the dashboard.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="RailPulse API",
    description="AI-Powered Railway Risk & Resource Allocation Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_FILE = Path("data/predictions.json")


def load_predictions():
    if not PREDICTIONS_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="Predictions not yet generated. Run `python model/predict.py` first."
        )
    with open(PREDICTIONS_FILE) as f:
        return json.load(f)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "RailPulse API",
        "version": "1.0.0",
        "status": "operational",
        "tagline": "Predict. Prioritize. Protect.",
        "endpoints": ["/api/summary", "/api/predictions", "/api/trains", "/api/stations", "/api/recommendations"],
    }


@app.get("/api/summary", tags=["Dashboard"])
def get_summary():
    """Returns headline stats for the top summary cards."""
    data = load_predictions()
    return {
        "generated_at": data["generated_at"],
        "date": data["date"],
        **data["summary"],
    }


@app.get("/api/predictions", tags=["Predictions"])
def get_all_predictions():
    """Returns complete predictions for all trains and coaches."""
    return load_predictions()


@app.get("/api/trains", tags=["Trains"])
def get_trains(sort_by: str = "risk"):
    """Returns trains sorted by risk score or departure time."""
    data = load_predictions()
    trains = data["trains"]
    if sort_by == "departure":
        trains = sorted(trains, key=lambda x: x["departs"])
    else:
        trains = sorted(trains, key=lambda x: x["aggregate_risk"], reverse=True)
    return {"trains": trains, "generated_at": data["generated_at"]}


@app.get("/api/trains/{train_id}", tags=["Trains"])
def get_train(train_id: str):
    """Returns detailed coach-level predictions for a specific train."""
    data = load_predictions()
    for train in data["trains"]:
        if train["train_id"] == train_id:
            return train
    raise HTTPException(status_code=404, detail=f"Train {train_id} not found")


@app.get("/api/stations", tags=["Stations"])
def get_stations():
    """Returns station congestion forecasts (6-hour timeline)."""
    data = load_predictions()
    return {
        "stations": data["stations"],
        "generated_at": data["generated_at"],
    }


@app.get("/api/recommendations", tags=["Recommendations"])
def get_recommendations(priority: str = None):
    """Returns AI-generated staff deployment recommendations."""
    data = load_predictions()
    recs = data["recommendations"]
    if priority:
        recs = [r for r in recs if r["priority"] == priority]
    return {
        "recommendations": recs,
        "total": len(recs),
        "generated_at": data["generated_at"],
    }


@app.post("/api/refresh", tags=["Admin"])
def refresh_predictions():
    """Triggers a fresh prediction run (re-runs predict.py)."""
    try:
        result = subprocess.run(
            ["python", "model/predict.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"status": "refreshed", "refreshed_at": datetime.now().isoformat()}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Prediction refresh timed out")


if __name__ == "__main__":
    import uvicorn
    print("🚂 Starting RailPulse API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
