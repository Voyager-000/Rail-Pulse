"""
RailPulse - FastAPI Backend with Live Simulation Engine
Serves risk predictions and real-time live simulation.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
import random
from datetime import datetime
from pathlib import Path

app = FastAPI(title="RailPulse API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PREDICTIONS_FILE = Path("data/predictions.json")
LIVE_STATE = None

STATION_COORDS = {
  'NDLS': [28.6139, 77.2090], 'BCT':  [18.9690, 72.8205],
  'HWH':  [22.5855, 88.3412], 'SBC':  [12.9781, 77.5695],
  'BPL':  [23.2599, 77.4126], 'SDAH': [22.5678, 88.3712],
  'BDTS': [19.0553, 72.8354], 'CSTM': [18.9398, 72.8354],
  'FZR':  [30.9304, 74.6186], 'RJPB': [25.5960, 85.1517],
  'TVC':  [8.4875,  76.9486], 'LTT':  [19.0683, 72.8906],
  'DBRG': [27.4728, 94.9120], 'ERS':  [9.9691,  76.2778],
  'VAR':  [25.3176, 82.9739], 'RTE':  [17.3850, 78.4867],
}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(req: LoginRequest):
    # For hackathon demo, accept 'password'
    if req.password == "password":
        return {"token": "secure-irctc-token-99x", "user": req.username.upper()}
    raise HTTPException(status_code=401, detail="Invalid credentials")

async def simulator_loop():
    global LIVE_STATE
    while not PREDICTIONS_FILE.exists():
        await asyncio.sleep(1)
    
    with open(PREDICTIONS_FILE) as f:
        LIVE_STATE = json.load(f)
    
    # Initialize physical coordinates for trains
    for train in LIVE_STATE["trains"]:
        parts = train["route"].split('→')
        orig = STATION_COORDS.get(parts[0])
        dest = STATION_COORDS.get(parts[1]) if len(parts) > 1 else None
        
        # Fallbacks for unknown stations
        if not orig: orig = [22.0 + random.uniform(-5, 5), 78.0 + random.uniform(-5, 5)]
        if not dest: dest = [22.0 + random.uniform(-5, 5), 78.0 + random.uniform(-5, 5)]
        
        progress = random.uniform(0.1, 0.9)
        train["lat"] = orig[0] + (dest[0] - orig[0]) * progress
        train["lon"] = orig[1] + (dest[1] - orig[1]) * progress
        train["target_lat"] = dest[0]
        train["target_lon"] = dest[1]
        train["orig_lat"] = orig[0]
        train["orig_lon"] = orig[1]

    while True:
        await asyncio.sleep(2.0) # Tick every 2 seconds
        
        # 1. Move trains
        for train in LIVE_STATE["trains"]:
            lat_diff = train["target_lat"] - train["lat"]
            lon_diff = train["target_lon"] - train["lon"]
            dist = (lat_diff**2 + lon_diff**2)**0.5
            
            if dist < 0.1:
                # Turn around
                train["target_lat"], train["orig_lat"] = train["orig_lat"], train["target_lat"]
                train["target_lon"], train["orig_lon"] = train["orig_lon"], train["target_lon"]
            else:
                speed = 0.05  # Move slowly
                train["lat"] += (lat_diff / dist) * speed
                train["lon"] += (lon_diff / dist) * speed

        # 2. Simulate Dynamic Boarding/Alighting (Random spikes)
        if random.random() < 0.4: # 40% chance every 2s
            t = random.choice(LIVE_STATE["trains"])
            c = random.choice(t["coaches"])
            
            # Fluctuate occupancy
            fluctuation = random.uniform(-0.05, 0.15)
            c["occupancy_ratio"] = max(0.1, min(1.8, c["occupancy_ratio"] + fluctuation))
            c["booked_seats"] = int(c["capacity"] * c["occupancy_ratio"])
            
            # Fast approximation of XGBoost logic for demo
            c["overcrowding_risk"] = min(0.99, max(0.01, (c["occupancy_ratio"] - 0.7) * 1.5))
            c["composite_risk"] = round(c["overcrowding_risk"] * 0.6 + c["ticketless_risk"] * 0.4, 3)
            
            if c["composite_risk"] >= 0.7: c["risk_level"] = "critical"
            elif c["composite_risk"] >= 0.45: c["risk_level"] = "high"
            elif c["composite_risk"] >= 0.25: c["risk_level"] = "medium"
            else: c["risk_level"] = "low"
            
            # Update train aggregate
            t["aggregate_risk"] = round(sum(coach["composite_risk"] for coach in t["coaches"]) / len(t["coaches"]), 3)
            t["critical_coaches"] = sum(1 for coach in t["coaches"] if coach["risk_level"] == "critical")
            
        LIVE_STATE["generated_at"] = datetime.now().isoformat()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulator_loop())

@app.get("/api/live", tags=["Live"])
def get_live_state():
    if not LIVE_STATE:
        raise HTTPException(status_code=503, detail="Simulator starting up...")
    
    # Recalculate summary live
    all_coaches = [c for t in LIVE_STATE["trains"] for c in t["coaches"]]
    LIVE_STATE["summary"]["critical_coaches"] = sum(1 for c in all_coaches if c["risk_level"] == "critical")
    LIVE_STATE["summary"]["high_risk_coaches"] = sum(1 for c in all_coaches if c["risk_level"] in ["critical", "high"])
    
    # Sort trains for leaderboard
    LIVE_STATE["trains"] = sorted(LIVE_STATE["trains"], key=lambda x: x["aggregate_risk"], reverse=True)
    return LIVE_STATE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
