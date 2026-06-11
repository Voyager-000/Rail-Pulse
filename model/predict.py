"""
RailPulse - Prediction Engine
Loads trained models and generates today's risk predictions for all coaches/trains/stations.
Outputs structured predictions.json for the dashboard.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta

print("🔮 RailPulse — Prediction Engine")
print("=" * 50)

# ─── Load Model Artifacts ─────────────────────────────────────────────────────
print("\n📂 Loading models...")
with open("model/overcrowd_model.pkl", "rb") as f:
    overcrowd_model = pickle.load(f)
with open("model/ticketless_model.pkl", "rb") as f:
    ticketless_model = pickle.load(f)
with open("model/scaler_oc.pkl", "rb") as f:
    scaler_oc = pickle.load(f)
with open("model/scaler_tl.pkl", "rb") as f:
    scaler_tl = pickle.load(f)
with open("model/model_meta.json") as f:
    meta = json.load(f)

OVERCROWD_FEATURES = meta["overcrowd_features"]
TICKETLESS_FEATURES = meta["ticketless_features"]
coach_type_map = meta["coach_type_map"]

print("   ✅ All models loaded")

# ─── Indian Railways Definitions ──────────────────────────────────────────────
real_trains_df = pd.read_csv("data/real_trains.csv", header=None, names=["id", "name"], dtype=str)
real_trains_df = real_trains_df[real_trains_df["id"].str.isnumeric() == True].dropna()
real_trains_df = real_trains_df[real_trains_df["name"].str.contains("Express|Rajdhani|Shatabdi|Duronto|Mail", case=False, na=False)]
real_trains_df = real_trains_df.sample(n=50, random_state=42).reset_index(drop=True)

TRAINS = []
for idx, row in real_trains_df.iterrows():
    words = row["name"].split()
    route = f"{words[0][:4].upper()}→{words[-2][:4].upper()}" if len(words) >= 3 else "VAR→RTE"
    # Generate consistent random departure time based on train ID
    np.random.seed(int(row["id"]))
    hour = np.random.randint(5, 23)
    minute = np.random.choice([0, 15, 30, 45])
    departs = f"{hour:02d}:{minute:02d}"
    popularity = np.random.uniform(0.65, 0.95)
    TRAINS.append({
        "id": row["id"],
        "name": row["name"].title(),
        "route": route,
        "departs": departs,
        "popularity": popularity
    })
np.random.seed() # reset seed


COACH_TYPES = {
    "SL": {"prefix": "S", "count": 10, "capacity": 72},
    "3A": {"prefix": "B", "count": 4,  "capacity": 64},
    "2A": {"prefix": "A", "count": 3,  "capacity": 46},
    "1A": {"prefix": "H", "count": 1,  "capacity": 18},
}

STATIONS = [
    {"code": "NDLS", "name": "New Delhi", "platforms": 16, "base_load": 0.85},
    {"code": "BCT",  "name": "Mumbai Central", "platforms": 8, "base_load": 0.78},
    {"code": "HWH",  "name": "Howrah Junction", "platforms": 23, "base_load": 0.80},
    {"code": "SBC",  "name": "KSR Bengaluru City", "platforms": 10, "base_load": 0.72},
    {"code": "BPL",  "name": "Bhopal Junction", "platforms": 6, "base_load": 0.65},
    {"code": "SDAH", "name": "Sealdah", "platforms": 13, "base_load": 0.75},
    {"code": "CSTM", "name": "Chhatrapati Shivaji Terminus", "platforms": 18, "base_load": 0.82},
    {"code": "TVC",  "name": "Thiruvananthapuram Central", "platforms": 5, "base_load": 0.60},
]

# ─── Determine Today's Context ────────────────────────────────────────────────
today = datetime.now()
today_str = today.strftime("%Y-%m-%d")
day_of_week = today.weekday()
is_weekend = int(day_of_week >= 5)

FESTIVAL_SURGE_DATES = {
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    "2026-01-24", "2026-01-25", "2026-01-26", "2026-01-27", "2026-01-28",
    "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05",
    "2026-10-18", "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22",
    "2026-12-23", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-27",
}
FESTIVAL_DATES = {"2026-01-14", "2026-01-26", "2026-03-03", "2026-10-20", "2026-12-25"}

is_holiday = int(today_str in FESTIVAL_DATES)
is_festival_surge = int(today_str in FESTIVAL_SURGE_DATES)
event_nearby = 0
if today.month == 6 and today.day <= 15:
    event_nearby = 1  # IPL season demo flag

print(f"\n📅 Generating predictions for: {today_str}")
print(f"   Day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]} | Weekend: {bool(is_weekend)} | Festival: {bool(is_holiday)}")

# ─── Generate Coach Predictions ───────────────────────────────────────────────
print("\n🔄 Running inference on all coaches...")

all_train_predictions = []

for train in TRAINS:
    coaches_out = []

    for ctype, cinfo in COACH_TYPES.items():
        for ci in range(1, cinfo["count"] + 1):
            coach_id = f"{cinfo['prefix']}{ci}"
            capacity = cinfo["capacity"]
            coach_type_enc = coach_type_map[ctype]

            # Simulate today's booking state
            base_occ = train["popularity"] * 0.72
            surge = 1.25 if is_holiday else (1.10 if is_weekend else 1.0)
            sl_boost = 1.15 if ctype == "SL" else 0.90
            occupancy_ratio = min(1.35, base_occ * surge * sl_boost + np.random.normal(0, 0.06))
            occupancy_ratio = max(0.1, occupancy_ratio)
            booked_seats = int(capacity * occupancy_ratio)
            overbooking_index = max(0, occupancy_ratio - 1.0)
            cancellation_rate = np.random.uniform(0.03, 0.15)
            historical_avg = train["popularity"] * 0.70
            route_popularity = train["popularity"]
            past_ticketless = max(0, int(np.random.normal(
                (0.12 if ctype == "SL" else 0.04) * capacity * (1.5 if is_festival_surge else 1.0), 2
            )))

            route_risk = route_popularity * historical_avg
            pressure_score = (
                occupancy_ratio * 0.4 +
                overbooking_index * 0.25 +
                event_nearby * 0.15 +
                is_holiday * 0.1 +
                is_festival_surge * 0.1
            )
            ticketless_pressure = (
                (past_ticketless / capacity) * 0.5 +
                occupancy_ratio * 0.3 +
                int(ctype == "SL") * 0.2
            )
            day_type_enc = 2 if is_holiday else (1 if is_weekend else 0)

            # ── Overcrowding Prediction ──
            oc_row = {
                "day_of_week": day_of_week, "is_holiday": is_holiday,
                "is_festival_surge": is_festival_surge, "is_weekend": is_weekend,
                "event_nearby": event_nearby, "event_pressure_score": event_nearby,
                "occupancy_ratio": occupancy_ratio, "overbooking_index": overbooking_index,
                "cancellation_rate": cancellation_rate,
                "historical_avg_occupancy": historical_avg, "route_popularity": route_popularity,
                "coach_type_enc": coach_type_enc, "coach_number": ci,
                "day_type_enc": day_type_enc, "route_risk": route_risk,
                "pressure_score": pressure_score,
            }
            oc_df = pd.DataFrame([oc_row])[OVERCROWD_FEATURES]
            oc_scaled = scaler_oc.transform(oc_df)
            overcrowd_prob = float(overcrowd_model.predict_proba(oc_scaled)[0][1])

            # ── Ticketless Prediction ──
            tl_row = {
                "day_of_week": day_of_week, "is_holiday": is_holiday,
                "is_festival_surge": is_festival_surge, "is_weekend": is_weekend,
                "event_nearby": event_nearby,
                "occupancy_ratio": occupancy_ratio, "overbooking_index": overbooking_index,
                "past_ticketless_incidents": past_ticketless,
                "cancellation_rate": cancellation_rate,
                "historical_avg_occupancy": historical_avg, "route_popularity": route_popularity,
                "coach_type_enc": coach_type_enc, "coach_number": ci,
                "ticketless_pressure": ticketless_pressure,
            }
            tl_df = pd.DataFrame([tl_row])[TICKETLESS_FEATURES]
            tl_scaled = scaler_tl.transform(tl_df)
            ticketless_prob = float(ticketless_model.predict_proba(tl_scaled)[0][1])

            # ── Risk Level ──
            composite = overcrowd_prob * 0.6 + ticketless_prob * 0.4
            if composite >= 0.7:
                risk_level = "critical"
            elif composite >= 0.45:
                risk_level = "high"
            elif composite >= 0.25:
                risk_level = "medium"
            else:
                risk_level = "low"

            coaches_out.append({
                "coach_id": coach_id,
                "coach_type": ctype,
                "capacity": capacity,
                "booked_seats": booked_seats,
                "occupancy_ratio": round(occupancy_ratio, 3),
                "overcrowding_risk": round(overcrowd_prob, 3),
                "ticketless_risk": round(ticketless_prob, 3),
                "composite_risk": round(composite, 3),
                "risk_level": risk_level,
                "risk_factors": {
                    "occupancy_pct": round(occupancy_ratio * 100, 1),
                    "holiday_weight": round(is_holiday * 30 + is_festival_surge * 15, 1),
                    "historical_incidents": past_ticketless,
                    "event_pressure": round(event_nearby * 20, 1),
                    "cancellation_surge": round(cancellation_rate * 100, 1),
                },
            })

    coaches_sorted = sorted(coaches_out, key=lambda x: x["composite_risk"], reverse=True)
    aggregate_risk = np.mean([c["composite_risk"] for c in coaches_sorted])
    critical_count = sum(1 for c in coaches_sorted if c["risk_level"] == "critical")

    all_train_predictions.append({
        "train_id": train["id"],
        "train_name": train["name"],
        "route": train["route"],
        "departs": train["departs"],
        "aggregate_risk": round(aggregate_risk, 3),
        "critical_coaches": critical_count,
        "coaches": coaches_sorted,
    })

# ─── Generate Station Congestion Forecast ─────────────────────────────────────
print("🏛️  Generating station congestion forecasts...")

station_forecasts = []
now_hour = today.hour

for station in STATIONS:
    timeline = []
    for h in range(6):
        hour = (now_hour + h) % 24
        # Simulate congestion based on rush hours (7-10am, 5-9pm)
        rush_factor = 1.0
        if 7 <= hour <= 10:
            rush_factor = 1.35
        elif 17 <= hour <= 21:
            rush_factor = 1.40
        elif 12 <= hour <= 14:
            rush_factor = 1.10

        base = station["base_load"]
        if is_holiday:
            base *= 1.3
        elif is_weekend:
            base *= 1.15

        congestion = min(1.0, base * rush_factor + np.random.normal(0, 0.04))
        congestion = max(0.1, congestion)

        timeline.append({
            "hour": f"{hour:02d}:00",
            "congestion": round(congestion, 3),
            "level": "critical" if congestion > 0.85 else ("high" if congestion > 0.65 else ("medium" if congestion > 0.45 else "low")),
        })

    station_forecasts.append({
        "station_code": station["code"],
        "station_name": station["name"],
        "platforms": station["platforms"],
        "current_congestion": timeline[0]["congestion"],
        "current_level": timeline[0]["level"],
        "timeline": timeline,
    })

# ─── Generate AI Recommendations ──────────────────────────────────────────────
print("🤖 Generating AI recommendations...")

recommendations = []
for train_pred in all_train_predictions:
    critical_coaches = [c for c in train_pred["coaches"] if c["risk_level"] == "critical"]
    high_coaches = [c for c in train_pred["coaches"] if c["risk_level"] == "high"]

    for coach in critical_coaches[:2]:  # Top 2 critical per train
        tte_count = 3 if coach["overcrowding_risk"] > 0.8 else 2
        recommendations.append({
            "priority": "critical",
            "type": "tte_deployment",
            "train_id": train_pred["train_id"],
            "train_name": train_pred["train_name"],
            "coach_id": coach["coach_id"],
            "action": f"Deploy {tte_count} TTEs to Coach {coach['coach_id']} of {train_pred['train_name']} ({train_pred['train_id']}) departing at {train_pred['departs']}",
            "confidence": round(coach["composite_risk"] * 100, 1),
            "reason": f"Overcrowding probability {coach['overcrowding_risk']*100:.0f}%, Ticketless risk {coach['ticketless_risk']*100:.0f}%",
        })

    if train_pred["aggregate_risk"] > 0.65 and high_coaches:
        recommendations.append({
            "priority": "high",
            "type": "security_check",
            "train_id": train_pred["train_id"],
            "train_name": train_pred["train_name"],
            "coach_id": high_coaches[0]["coach_id"],
            "action": f"Conduct ticket verification sweep in coaches {', '.join([c['coach_id'] for c in high_coaches[:3]])} of {train_pred['train_name']}",
            "confidence": round(train_pred["aggregate_risk"] * 100, 1),
            "reason": f"Train aggregate risk {train_pred['aggregate_risk']*100:.0f}% — multiple high-risk coaches flagged",
        })

for station in station_forecasts:
    if station["current_level"] in ["critical", "high"]:
        recommendations.append({
            "priority": station["current_level"],
            "type": "station_management",
            "station_code": station["station_code"],
            "station_name": station["station_name"],
            "action": f"Increase crowd management staff at {station['station_name']} ({station['station_code']}) — congestion expected in next 2 hours",
            "confidence": round(station["current_congestion"] * 100, 1),
            "reason": f"Predicted congestion index: {station['current_congestion']*100:.0f}%",
        })

recommendations.sort(key=lambda x: ({"critical": 0, "high": 1, "medium": 2, "low": 3}[x["priority"]], -x["confidence"]))

# ─── Summary Stats ────────────────────────────────────────────────────────────
all_coaches_flat = [c for t in all_train_predictions for c in t["coaches"]]
high_risk_coaches = sum(1 for c in all_coaches_flat if c["risk_level"] in ["critical", "high"])
critical_coaches = sum(1 for c in all_coaches_flat if c["risk_level"] == "critical")
medium_risk_trains = sum(1 for t in all_train_predictions if 0.35 <= t["aggregate_risk"] < 0.65)
high_risk_trains = sum(1 for t in all_train_predictions if t["aggregate_risk"] >= 0.65)
staff_recommended = sum(
    3 if r.get("type") == "tte_deployment" and r["priority"] == "critical" else
    2 if r.get("type") == "tte_deployment" else
    4 if r.get("type") == "station_management" else 1
    for r in recommendations
)

summary = {
    "total_trains": len(TRAINS),
    "total_coaches": len(all_coaches_flat),
    "critical_coaches": critical_coaches,
    "high_risk_coaches": high_risk_coaches,
    "high_risk_trains": high_risk_trains,
    "medium_risk_trains": medium_risk_trains,
    "stations_monitored": len(STATIONS),
    "staff_recommended": min(staff_recommended, 50),
    "total_recommendations": len(recommendations),
    "avg_risk_score": round(np.mean([t["aggregate_risk"] for t in all_train_predictions]), 3),
    "model_accuracy": {
        "overcrowding_auc": meta["overcrowd_auc"],
        "ticketless_auc": meta["ticketless_auc"],
    },
}

# ─── Write Output ─────────────────────────────────────────────────────────────
output = {
    "generated_at": today.isoformat(),
    "date": today_str,
    "summary": summary,
    "trains": sorted(all_train_predictions, key=lambda x: x["aggregate_risk"], reverse=True),
    "stations": sorted(station_forecasts, key=lambda x: x["current_congestion"], reverse=True),
    "recommendations": recommendations[:20],
}

os.makedirs("data", exist_ok=True)
with open("data/predictions.json", "w") as f:
    json.dump(output, f, indent=2)

with open("data/predictions.js", "w") as f:
    f.write(f"const FALLBACK_DATA = {json.dumps(output)};")

print(f"\n✅ Predictions saved → data/predictions.json")
print(f"\n📊 Summary:")
print(f"   🔴 Critical coaches:    {critical_coaches}")
print(f"   🟠 High-risk coaches:   {high_risk_coaches}")
print(f"   🚂 High-risk trains:    {high_risk_trains}")
print(f"   🏛️  Stations monitored:  {len(STATIONS)}")
print(f"   👥 Staff recommended:   {staff_recommended}")
print(f"   📋 Recommendations:     {len(recommendations)}")
print(f"\n🎉 RailPulse prediction engine complete!")
