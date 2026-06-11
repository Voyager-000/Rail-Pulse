# RailPulse 🚂 — AI-Powered Railway Risk & Resource Allocation

> **Predict. Prioritize. Protect.**

Unlike solutions requiring passengers to install a new app, **RailPulse works entirely on existing railway booking and operational data**. Passengers don't change their behavior — railway staff simply get smarter, AI-driven directions.

---

## 🎯 Problem

- Millions travel daily on Indian Railways
- Limited TTEs and inspection staff
- Overcrowding and ticketless travel are discovered *after* they escalate
- Current systems are **reactive**, not proactive

## 💡 Solution

RailPulse predicts which coaches, trains, and stations need attention **before** problems escalate — using only data railway systems already collect.

## 📊 Sample Output

| Entity | Prediction |
|--------|-----------|
| Coach S3, Train 12951 | **87% overcrowding probability** |
| Train 12301 | **High ticketless passenger likelihood** |
| NDLS Station | **Peak congestion in next 2 hours** |

---

## 🏗️ Architecture

```
Booking Data → Feature Engineering → XGBoost Model → Risk Scores → Dashboard
                                                    ↓
                                          FastAPI REST API
```

### Stack
| Layer | Tech |
|-------|------|
| ML Model | Python, XGBoost, scikit-learn, Pandas |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML5, Vanilla JS, Chart.js, CSS (Glassmorphism) |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install pandas numpy xgboost scikit-learn fastapi uvicorn
```

### 2. Generate synthetic training data
```bash
python data/generate_data.py
```
→ Creates `data/train_data.csv` (~50,000 booking records across 20 trains)

### 3. Train the ML models
```bash
python model/train_model.py
```
→ Trains XGBoost classifiers for overcrowding + ticketless risk  
→ Saves model artifacts to `model/`

### 4. Run predictions
```bash
python model/predict.py
```
→ Generates today's risk scores → `data/predictions.json`

### 5. Open the dashboard
```bash
# Option A: Open directly (offline mode)
start dashboard/index.html

# Option B: With live API
cd api && uvicorn main:app --reload --port 8000
# Then open dashboard/index.html in your browser
```

---

## 📁 Project Structure

```
rail/
├── data/
│   ├── generate_data.py      # Synthetic data generator
│   ├── train_data.csv        # Generated training data
│   └── predictions.json      # Generated predictions
├── model/
│   ├── train_model.py        # XGBoost training pipeline
│   ├── predict.py            # Inference engine
│   ├── overcrowd_model.pkl   # Trained model artifacts
│   └── model_meta.json       # Feature config & AUC scores
├── api/
│   ├── main.py               # FastAPI REST server
│   └── requirements.txt
└── dashboard/
    ├── index.html            # Main dashboard
    ├── style.css             # Dark glassmorphism theme
    └── app.js                # Charts, heatmap, live refresh
```

---

## 🤖 Model Details

**Features used:**
- Booking occupancy ratio, overbooking index
- Day of week, weekend/holiday flags
- Indian festival calendar (Diwali, Holi, Eid, etc.)
- Historical ticketless incidents per coach
- Route popularity, cancellation rate, event proximity

**Outputs:**
- `overcrowding_risk`: 0–1 probability per coach
- `ticketless_risk`: 0–1 probability per coach
- `composite_risk`: weighted combination → risk level (low/medium/high/critical)

**Performance:** AUC > 0.85 on both classifiers (see `model/model_meta.json`)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/summary` | Headline stats |
| GET | `/api/predictions` | All coach/train risk scores |
| GET | `/api/trains` | Train leaderboard |
| GET | `/api/stations` | Station congestion (6-hr forecast) |
| GET | `/api/recommendations` | AI deployment directives |
| POST | `/api/refresh` | Re-run predictions |

---

## 🌍 Why This Works at Scale

- ✅ **No passenger app required** — uses existing IRCTC booking data
- ✅ **No behavior change** — staff get AI-driven directions, passengers board normally  
- ✅ **Works offline** — dashboard has embedded fallback JSON for demos
- ✅ **Production-ready model** — swap synthetic data with real IRCTC data, retrain, deploy

---

*Built for the RailPulse Hackathon 2026*
