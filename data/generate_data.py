"""
RailPulse - Synthetic Data Generator
Generates realistic Indian Railways booking records for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ─── Indian Railways Train Definitions ────────────────────────────────────────
TRAINS = [
    {"id": "12951", "name": "Mumbai Rajdhani", "route": "NDLS-BCT", "popularity": 0.92},
    {"id": "12952", "name": "Mumbai Rajdhani (Return)", "route": "BCT-NDLS", "popularity": 0.88},
    {"id": "12301", "name": "Howrah Rajdhani", "route": "NDLS-HWH", "popularity": 0.85},
    {"id": "12302", "name": "Howrah Rajdhani (Return)", "route": "HWH-NDLS", "popularity": 0.83},
    {"id": "22691", "name": "Rajdhani Express", "route": "SBC-NDLS", "popularity": 0.78},
    {"id": "12627", "name": "Karnataka Express", "route": "SBC-NDLS", "popularity": 0.72},
    {"id": "12628", "name": "Karnataka Express (Return)", "route": "NDLS-SBC", "popularity": 0.70},
    {"id": "12001", "name": "Bhopal Shatabdi", "route": "NDLS-BPL", "popularity": 0.81},
    {"id": "12002", "name": "Bhopal Shatabdi (Return)", "route": "BPL-NDLS", "popularity": 0.79},
    {"id": "12259", "name": "Sealdah Duronto", "route": "SDAH-NDLS", "popularity": 0.74},
    {"id": "12260", "name": "Sealdah Duronto (Return)", "route": "NDLS-SDAH", "popularity": 0.71},
    {"id": "19019", "name": "Dehradun Express", "route": "BDTS-DDN", "popularity": 0.65},
    {"id": "12137", "name": "Punjab Mail", "route": "CSTM-FZR", "popularity": 0.68},
    {"id": "12138", "name": "Punjab Mail (Return)", "route": "FZR-CSTM", "popularity": 0.66},
    {"id": "12393", "name": "Sampoorna Kranti", "route": "RJPB-NDLS", "popularity": 0.75},
    {"id": "12394", "name": "Sampoorna Kranti (Return)", "route": "NDLS-RJPB", "popularity": 0.73},
    {"id": "12223", "name": "Duronto Express", "route": "LTT-ERS", "popularity": 0.69},
    {"id": "12431", "name": "Trivandrum Rajdhani", "route": "NDLS-TVC", "popularity": 0.77},
    {"id": "12432", "name": "Trivandrum Rajdhani (Return)", "route": "TVC-NDLS", "popularity": 0.75},
    {"id": "12561", "name": "Swatantra Senani Express", "route": "DBRG-NDLS", "popularity": 0.62},
]

# ─── Coach Layouts ────────────────────────────────────────────────────────────
COACH_TYPES = {
    "SL": {"prefix": "S", "count": 10, "capacity": 72},   # Sleeper
    "3A": {"prefix": "B", "count": 4,  "capacity": 64},   # 3-Tier AC
    "2A": {"prefix": "A", "count": 3,  "capacity": 46},   # 2-Tier AC
    "1A": {"prefix": "H", "count": 1,  "capacity": 18},   # First AC
}

def get_coaches():
    coaches = []
    for ctype, info in COACH_TYPES.items():
        for i in range(1, info["count"] + 1):
            coaches.append({
                "coach_id": f"{info['prefix']}{i}",
                "coach_type": ctype,
                "capacity": info["capacity"],
                "is_sleeper": ctype == "SL",
            })
    return coaches

COACHES = get_coaches()

# ─── Indian Festival Calendar (2025–2026) ─────────────────────────────────────
FESTIVALS = [
    # 2025
    "2025-01-14",  # Makar Sankranti
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Maha Shivratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Eid ul-Fitr
    "2025-04-14",  # Dr. Ambedkar Jayanti / Baisakhi
    "2025-04-18",  # Good Friday
    "2025-05-12",  # Buddha Purnima
    "2025-06-07",  # Eid ul-Adha
    "2025-08-15",  # Independence Day
    "2025-08-16",  # Janmashtami
    "2025-09-04",  # Ganesh Chaturthi (approx)
    "2025-09-29",  # Navratri (approx)
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-12",  # Dussehra (approx)
    "2025-10-20",  # Diwali (approx)
    "2025-10-21",  # Diwali
    "2025-11-05",  # Bhai Dooj
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-14",  # Makar Sankranti
    "2026-01-26",  # Republic Day
    "2026-03-03",  # Holi
    "2026-03-20",  # Eid ul-Fitr
    "2026-04-02",  # Good Friday
    "2026-08-15",  # Independence Day
    "2026-10-20",  # Diwali (approx)
    "2026-12-25",  # Christmas
]

FESTIVAL_DATES = set(FESTIVALS)

# Dates near festivals also see surge (travel 2–3 days before/after)
FESTIVAL_SURGE_DATES = set()
for f in FESTIVALS:
    fd = datetime.strptime(f, "%Y-%m-%d")
    for delta in range(-3, 4):
        FESTIVAL_SURGE_DATES.add((fd + timedelta(days=delta)).strftime("%Y-%m-%d"))

# ─── Major Events ─────────────────────────────────────────────────────────────
EVENTS = {
    "2025-03-14": "IPL Season Start",
    "2025-04-18": "Good Friday Weekend",
    "2025-08-15": "Independence Day",
    "2025-10-21": "Diwali Peak",
    "2026-01-26": "Republic Day",
}

# ─── Historical Ticketless Incident Rates by Coach Type ───────────────────────
TICKETLESS_BASE = {
    "SL": 0.18,   # Sleeper most affected
    "3A": 0.07,
    "2A": 0.03,
    "1A": 0.01,
}

# ─── Data Generation ──────────────────────────────────────────────────────────
def generate_records(n_days=365, start_date="2025-01-01"):
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    for day_offset in range(n_days):
        date = start + timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        day_of_week = date.weekday()  # 0=Mon, 6=Sun

        is_festival = date_str in FESTIVAL_DATES
        is_festival_surge = date_str in FESTIVAL_SURGE_DATES
        is_weekend = day_of_week >= 5
        has_event = date_str in EVENTS
        event_name = EVENTS.get(date_str, "")

        for train in TRAINS:
            for coach in COACHES:
                capacity = coach["capacity"]

                # ── Base occupancy influenced by train popularity ──
                base_occ = train["popularity"] * 0.72

                # ── Seasonal & festival modifiers ──
                if is_festival:
                    occ_modifier = np.random.uniform(1.15, 1.45)
                elif is_festival_surge:
                    occ_modifier = np.random.uniform(1.05, 1.30)
                elif is_weekend:
                    occ_modifier = np.random.uniform(1.02, 1.18)
                elif has_event:
                    occ_modifier = np.random.uniform(1.10, 1.25)
                else:
                    occ_modifier = np.random.uniform(0.85, 1.05)

                # ── Coach-level variance (SL more crowded) ──
                coach_modifier = 1.0
                if coach["coach_type"] == "SL":
                    coach_modifier = np.random.uniform(1.05, 1.25)
                elif coach["coach_type"] == "3A":
                    coach_modifier = np.random.uniform(0.90, 1.10)
                elif coach["coach_type"] in ["2A", "1A"]:
                    coach_modifier = np.random.uniform(0.75, 0.95)

                booked_seats = int(capacity * base_occ * occ_modifier * coach_modifier)
                booked_seats = min(booked_seats, int(capacity * 1.30))  # cap at 130%
                booked_seats = max(booked_seats, 0)

                occupancy_ratio = booked_seats / capacity

                # ── Cancellation rate ──
                cancellation_rate = np.random.uniform(0.02, 0.18)
                if is_festival:
                    cancellation_rate *= 0.5   # fewer cancellations on festival days

                # ── Ticketless incident rate ──
                base_ticketless = TICKETLESS_BASE[coach["coach_type"]]
                ticketless_modifier = 1.0
                if is_festival_surge:
                    ticketless_modifier = np.random.uniform(1.5, 2.5)
                elif occupancy_ratio > 1.0:
                    ticketless_modifier = np.random.uniform(1.3, 2.0)

                past_ticketless_incidents = max(
                    0,
                    int(np.random.normal(base_ticketless * ticketless_modifier * capacity, 2))
                )

                # ── Historical average occupancy ──
                hist_avg = train["popularity"] * 0.70 + np.random.uniform(-0.05, 0.05)

                # ── Route popularity ──
                route_popularity = train["popularity"] + np.random.uniform(-0.05, 0.05)

                # ── Engineered features ──
                overbooking_index = max(0, occupancy_ratio - 1.0)
                event_pressure = 1 if (is_festival or has_event) else 0
                day_type = "weekend" if is_weekend else ("festival" if is_festival else "weekday")

                # ── Target labels ──
                # Overcrowding: probabilistic label based on occupancy
                overcrowd_prob = min(1.0, max(0.0,
                    (occupancy_ratio - 0.65) * 1.5 +
                    (0.3 if is_festival else 0) +
                    (0.15 if is_weekend else 0) +
                    np.random.normal(0, 0.08)
                ))
                overcrowding_label = int(overcrowd_prob > 0.5)

                # Ticketless: probabilistic label
                ticketless_prob = min(1.0, max(0.0,
                    (past_ticketless_incidents / capacity) * 1.5 +
                    (0.25 if coach["coach_type"] == "SL" else 0) +
                    (0.2 if occupancy_ratio > 0.95 else 0) +
                    np.random.normal(0, 0.06)
                ))
                ticketless_label = int(ticketless_prob > 0.4)

                records.append({
                    "date": date_str,
                    "day_of_week": day_of_week,
                    "is_holiday": int(is_festival),
                    "is_festival_surge": int(is_festival_surge),
                    "is_weekend": int(is_weekend),
                    "event_nearby": int(has_event),
                    "event_name": event_name,
                    "day_type": day_type,
                    "train_id": train["id"],
                    "train_name": train["name"],
                    "route": train["route"],
                    "route_popularity": round(route_popularity, 3),
                    "coach_id": coach["coach_id"],
                    "coach_type": coach["coach_type"],
                    "total_seats": capacity,
                    "booked_seats": booked_seats,
                    "occupancy_ratio": round(occupancy_ratio, 4),
                    "overbooking_index": round(overbooking_index, 4),
                    "cancellation_rate": round(cancellation_rate, 4),
                    "past_ticketless_incidents": past_ticketless_incidents,
                    "historical_avg_occupancy": round(hist_avg, 4),
                    "event_pressure_score": event_pressure,
                    "overcrowding_label": overcrowding_label,
                    "ticketless_label": ticketless_label,
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("🚂 RailPulse — Generating synthetic training data...")
    print(f"   Trains: {len(TRAINS)} | Coaches per train: {len(COACHES)}")

    df = generate_records(n_days=365)

    os.makedirs("data", exist_ok=True)
    out_path = "data/train_data.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Generated {len(df):,} records → {out_path}")
    print(f"\n📊 Dataset Overview:")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"   Overcrowding rate: {df['overcrowding_label'].mean():.1%}")
    print(f"   Ticketless rate:   {df['ticketless_label'].mean():.1%}")
    print(f"   Avg occupancy:     {df['occupancy_ratio'].mean():.1%}")
    print(f"   Columns: {list(df.columns)}")
