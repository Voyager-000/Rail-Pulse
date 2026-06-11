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
# Station code lookup — used to build real origin→destination routes from train names
STATION_CODE_MAP = {
    "DELHI": "NDLS", "NEW": "NDLS", "NIZAMUDDIN": "NZM",
    "MUMBAI": "BCT",  "BOMBAY": "BCT",   "BANDRA": "BDTS",
    "HOWRAH": "HWH",  "KOLKATA": "SDAH",  "SEALDAH": "SDAH",
    "BENGALURU": "SBC", "BANGALORE": "SBC", "CHENNAI": "MAS",
    "HYDERABAD": "HYB", "SECUNDERABAD": "SC", "AHMEDABAD": "ADI",
    "PUNE": "PUNE",  "NAGPUR": "NGP",   "BHOPAL": "BPL",
    "PATNA": "PNBE", "LUCKNOW": "LKO",   "KANPUR": "CNB",
    "JAIPUR": "JP",  "VARANASI": "BSB",  "ALLAHABAD": "ALD",
    "PRAYAGRAJ": "PRYJ", "GUWAHATI": "GHY", "DIBRUGARH": "DBRG",
    "TRIVANDRUM": "TVC", "THIRUVANANTHAPURAM": "TVC", "ERNAKULAM": "ERS",
    "COIMBATORE": "CBE", "MADURAI": "MDU",  "VISAKHAPATNAM": "VSKP",
    "VIJAYAWADA": "BZA", "KOCHI": "ERS",   "SURAT": "ST", "RAJKOT": "RJT",
    "AGRA": "AGC",   "MATHURA": "MTJ",   "AMRITSAR": "ASR", "FIROZPUR": "FZR",
    "JAMMU": "JAT",  "DEHRADUN": "DDN",  "HARIDWAR": "HW",  "MEERUT": "MTC",
    "GORAKHPUR": "GKP", "RAIPUR": "R",   "RANCHI": "RNC",  "BHUBANESWAR": "BBS",
}

HIGH_POPULARITY_ROUTES = [
    "NDLS", "BCT", "HWH", "SBC", "MAS", "HYB", "ADI", "PNBE", "LKO", "NZM"
]

real_trains_df = pd.read_csv("data/real_trains.csv", header=None, names=["id", "name"], dtype=str)
real_trains_df = real_trains_df[real_trains_df["id"].str.isnumeric() == True].dropna()
real_trains_df = real_trains_df[real_trains_df["name"].str.contains("Express|Rajdhani|Shatabdi|Duronto|Mail", case=False, na=False)]
real_trains_df = real_trains_df.sample(n=50, random_state=42).reset_index(drop=True)

TRAINS = []
for idx, row in real_trains_df.iterrows():
    # Build a proper route using real station codes from the train name
    words = row["name"].upper().split()
    orig_code, dest_code = None, None
    for word in words:
        if not orig_code and word in STATION_CODE_MAP:
            orig_code = STATION_CODE_MAP[word]
        elif orig_code and word in STATION_CODE_MAP:
            dest_code = STATION_CODE_MAP[word]
            break
    # Fallback: try partial match on word sequences
    if not orig_code:
        orig_code = next((STATION_CODE_MAP[k] for k in STATION_CODE_MAP if any(w in k for w in words[:3])), "NDLS")
    if not dest_code:
        dest_code = next((STATION_CODE_MAP[k] for k in STATION_CODE_MAP if any(w in k for w in words[-3:])), "BCT")
    if orig_code == dest_code:
        dest_code = "HWH"
    route = f"{orig_code}→{dest_code}"

    # Route-based popularity — Delhi/Mumbai routes are busier
    high_pop = any(c in [orig_code, dest_code] for c in HIGH_POPULARITY_ROUTES)
    np.random.seed(int(row["id"]))
    hour = np.random.randint(5, 23)
    minute = np.random.choice([0, 10, 15, 20, 30, 45, 55])
    departs = f"{hour:02d}:{minute:02d}"
    popularity = np.random.uniform(0.78, 0.98) if high_pop else np.random.uniform(0.60, 0.85)

    # Real trains are often delayed — simulate delay in minutes
    delay_prob = 0.72  # 72% of Indian trains run late
    delay_minutes = int(np.random.choice(
        [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180],
        p=[0.28, 0.10, 0.10, 0.10, 0.10, 0.10, 0.07, 0.07, 0.04, 0.03, 0.01]
    ))

    TRAINS.append({
        "id": row["id"],
        "name": row["name"].title(),
        "route": route,
        "departs": departs,
        "delay_minutes": delay_minutes,
        "popularity": popularity
    })

COACH_TYPES = {
    "GEN": {"prefix": "GEN", "count": 4,  "capacity": 90},   # General / Unreserved — most overcrowded
    "SL":  {"prefix": "S",   "count": 10, "capacity": 72},   # Sleeper Class
    "3A":  {"prefix": "B",   "count": 4,  "capacity": 64},   # AC 3-Tier
    "2A":  {"prefix": "A",   "count": 2,  "capacity": 46},   # AC 2-Tier
    "1A":  {"prefix": "H",   "count": 1,  "capacity": 18},   # AC First Class
}

STATIONS = [
    {"code": "NDLS", "name": "New Delhi",                    "platforms": 16, "base_load": 0.88},
    {"code": "BCT",  "name": "Mumbai Central",               "platforms": 8,  "base_load": 0.82},
    {"code": "HWH",  "name": "Howrah Junction",              "platforms": 23, "base_load": 0.85},
    {"code": "SBC",  "name": "KSR Bengaluru City",           "platforms": 10, "base_load": 0.75},
    {"code": "BPL",  "name": "Bhopal Junction",              "platforms": 6,  "base_load": 0.68},
    {"code": "SDAH", "name": "Sealdah",                      "platforms": 13, "base_load": 0.79},
    {"code": "CSTM", "name": "Chhatrapati Shivaji Terminus", "platforms": 18, "base_load": 0.84},
    {"code": "TVC",  "name": "Thiruvananthapuram Central",   "platforms": 5,  "base_load": 0.62},
]

# ─── Determine Today's Context ────────────────────────────────────────────────
today = datetime.now()
today_str = today.strftime("%Y-%m-%d")
day_of_week = today.weekday()
is_weekend = int(day_of_week >= 5)

# ─── Festival & Event Calendar 2026 ──────────────────────────────────────────
FESTIVAL_SURGE_DATES = {
    # Makar Sankranti/Pongal (Jan 13-15)
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16",
    # Republic Day (Jan 26)
    "2026-01-24", "2026-01-25", "2026-01-26", "2026-01-27", "2026-01-28",
    # Holi (Mar 3)
    "2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05",
    # Ram Navami (Mar 27)
    "2026-03-25", "2026-03-26", "2026-03-27",
    # Eid ul-Fitr (Apr, approximate)
    "2026-04-19", "2026-04-20", "2026-04-21", "2026-04-22",
    # Eid ul-Adha (Jun 27)
    "2026-06-25", "2026-06-26", "2026-06-27", "2026-06-28",
    # Independence Day (Aug 15)
    "2026-08-13", "2026-08-14", "2026-08-15", "2026-08-16",
    # Janmashtami (Aug 23)
    "2026-08-22", "2026-08-23", "2026-08-24",
    # Ganesh Chaturthi (Sep 8)
    "2026-09-07", "2026-09-08", "2026-09-09",
    # Navratri / Dussehra (Oct)
    "2026-10-09", "2026-10-10", "2026-10-11", "2026-10-18", "2026-10-19", "2026-10-20",
    # Diwali (Oct 29)
    "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31",
    # Chhath Puja (Nov 1) — heaviest train traffic event in India
    "2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05",
    # Christmas / New Year
    "2026-12-23", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-27",
    "2026-12-30", "2026-12-31",
}
FESTIVAL_DATES = {
    "2026-01-14", "2026-01-26", "2026-03-03", "2026-03-27",
    "2026-08-15", "2026-10-20", "2026-10-29", "2026-11-02", "2026-12-25",
}

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
            coach_type_enc = coach_type_map.get(ctype, coach_type_map["SL"])  # GEN maps to SL encoder

            # GEN coaches are unreserved — always extremely overcrowded
            is_gen = (ctype == "GEN")

            # Simulate today's booking state
            base_occ = train["popularity"] * 0.95
            surge = 1.4 if is_holiday else (1.25 if is_weekend else 1.15)
            if is_gen:
                # GEN coaches routinely hit 200-300% occupancy on popular routes
                sl_boost = 1.80 + np.random.uniform(0, 0.5)
            elif ctype == "SL":
                sl_boost = 1.25
            else:
                sl_boost = 0.95
            occupancy_ratio = min(2.50 if is_gen else 1.50, base_occ * surge * sl_boost + np.random.normal(0, 0.08))
            occupancy_ratio = max(0.5 if is_gen else 0.1, occupancy_ratio)
            booked_seats = int(capacity * min(occupancy_ratio, 1.0))  # Physical seats limited to capacity
            overbooking_index = max(0, occupancy_ratio - 1.0)
            cancellation_rate = np.random.uniform(0.02, 0.10)
            historical_avg = train["popularity"] * 0.85
            route_popularity = train["popularity"]
            # GEN and SL coaches have much higher ticketless passengers
            ticketless_base = 0.35 if is_gen else (0.20 if ctype == "SL" else 0.08)
            past_ticketless = max(0, int(np.random.normal(
                ticketless_base * capacity * (1.5 if is_festival_surge else 1.2), 2
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
    # Delay increases composite risk (delayed trains attract crowd spillovers)
    delay_min = train.get("delay_minutes", 0)
    delay_status = "On Time" if delay_min == 0 else (
        f"Delayed {delay_min} min" if delay_min < 60 else f"Delayed {delay_min // 60}h {delay_min % 60}m"
    )

    all_train_predictions.append({
        "train_id": train["id"],
        "train_name": train["name"],
        "route": train["route"],
        "departs": train["departs"],
        "delay_minutes": delay_min,
        "delay_status": delay_status,
        "aggregate_risk": round(min(1.0, aggregate_risk + delay_min * 0.0008), 3),
        "critical_coaches": critical_count,
        "coaches": coaches_sorted,
    })

# ─── Generate Station Congestion Forecast ─────────────────────────────────────
print("🏛️  Generating station congestion forecasts...")

station_forecasts = []
now_hour = today.hour

for station in STATIONS:
    timeline = []
    for h in range(12):  # 12-hour forecast window
        hour = (now_hour + h) % 24
        # Realistic Indian station rush hour patterns:
        # Major peaks: 6-10am (morning rush), 5-9pm (evening rush)
        # Secondary peak: 12-2pm (afternoon)
        if 6 <= hour <= 9:
            rush_factor = 1.0 + 0.5 * (1 - abs(hour - 7.5) / 1.5)  # Peak at 7:30am
        elif 17 <= hour <= 21:
            rush_factor = 1.0 + 0.55 * (1 - abs(hour - 19) / 2.0)  # Peak at 7pm
        elif 12 <= hour <= 14:
            rush_factor = 1.15
        elif 1 <= hour <= 4:
            rush_factor = 0.4  # Dead of night — very low traffic
        elif 22 <= hour or hour == 0:
            rush_factor = 0.7
        else:
            rush_factor = 0.9

        base = station["base_load"]
        if is_holiday:
            base *= 1.35
        elif is_festival_surge:
            base *= 1.25
        elif is_weekend:
            base *= 1.15

        congestion = min(1.0, base * rush_factor + np.random.normal(0, 0.03))
        congestion = max(0.05, congestion)

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

    for coach in critical_coaches:  # All critical coaches, not just top 2
        # Real Indian Railways deploys 2-4 TTEs per high-risk coach
        tte_count = 4 if coach["overcrowding_risk"] > 0.85 else 3 if coach["overcrowding_risk"] > 0.70 else 2
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

    for coach in high_coaches:  # All high-risk coaches too
        tte_count = 2 if coach["ticketless_risk"] > 0.6 else 1
        recommendations.append({
            "priority": "high",
            "type": "tte_deployment",
            "train_id": train_pred["train_id"],
            "train_name": train_pred["train_name"],
            "coach_id": coach["coach_id"],
            "action": f"Deploy {tte_count} TTE(s) to Coach {coach['coach_id']} of {train_pred['train_name']} ({train_pred['train_id']}) — high ticketless risk",
            "confidence": round(coach["composite_risk"] * 100, 1),
            "reason": f"Ticketless risk {coach['ticketless_risk']*100:.0f}%, Overcrowding risk {coach['overcrowding_risk']*100:.0f}%",
        })

    if train_pred["aggregate_risk"] >= 0.65 and high_coaches:
        recommendations.append({
            "priority": "high",
            "type": "security_check",
            "train_id": train_pred["train_id"],
            "train_name": train_pred["train_name"],
            "coach_id": high_coaches[0]["coach_id"],
            "action": f"Conduct ticket verification sweep in coaches {', '.join([c['coach_id'] for c in high_coaches[:4]])} of {train_pred['train_name']}",
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
# Staff = actual TTE headcount: sum TTEs per coach recommendation
staff_recommended = sum(
    4 if r.get("type") == "tte_deployment" and r["priority"] == "critical" and r["confidence"] >= 85 else
    3 if r.get("type") == "tte_deployment" and r["priority"] == "critical" else
    2 if r.get("type") == "tte_deployment" and r["priority"] == "high" else
    6 if r.get("type") == "station_management" and r["priority"] == "critical" else
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
    "staff_recommended": staff_recommended,  # Real count, no arbitrary cap
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
    "recommendations": recommendations,  # All recommendations, no slice limit
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
