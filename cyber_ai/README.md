# NeuroSent — AI-Powered Cyber Threat Intelligence

> Real-time cyber threat detection using Machine Learning (Isolation Forest, Random Forest, KMeans) with a live operational dashboard.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)

---

## Features

- **Real-Time Threat Detection** — Classifies DDoS, brute force, port scans, SQL injection, malware C2, and zero-day attacks
- **3 ML Models Working Together** — Isolation Forest (anomaly detection) + Random Forest (classification) + KMeans (behavior clustering)
- **SHAP-Style Explainability** — Top contributing features shown for every alert
- **WebSocket Live Feed** — Continuous ML predictions streamed to the dashboard
- **Interactive Dashboard** — D3.js world map with animated attack arcs, live metrics, threat feed

---

## Architecture

```
┌─────────────┐    WebSocket     ┌──────────────────┐
│  Dashboard   │◄───────────────►│   FastAPI Server  │
│  (Browser)   │    REST API     │   (main.py)       │
└─────────────┘                  └────────┬─────────┘
                                          │
                               ┌──────────▼──────────┐
                               │  PredictionEngine    │
                               │  (models/predict.py) │
                               └──────────┬──────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
             ┌──────▼──────┐     ┌───────▼───────┐     ┌──────▼──────┐
             │  Isolation   │     │  Random Forest │     │   KMeans    │
             │  Forest      │     │  Classifier    │     │  Clustering │
             │  (anomaly)   │     │  (threats)     │     │  (behavior) │
             └─────────────┘     └───────────────┘     └─────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd cyber_ai
pip install -r requirements.txt
```

### 2. Generate Synthetic Training Data

```bash
python data/generate_data.py
```

This creates `data/sample_traffic.csv` with 10,000 rows of network traffic (60% normal, 40% split across 6 attack types).

### 3. Train ML Models

```bash
python models/train.py
```

Trains all 3 models and saves them to `models/saved/`. You'll see a full classification report with accuracy metrics.

### 4. Start the Server

```bash
python main.py
```

### 5. Open the Dashboard

Navigate to **http://localhost:8000** in your browser.

The dashboard is now fully live with real ML predictions streaming in real time.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the dashboard |
| `POST` | `/api/predict` | Single traffic sample prediction |
| `POST` | `/api/predict/batch` | Batch prediction |
| `GET` | `/api/stats` | Live dashboard statistics |
| `POST` | `/api/simulate/attack` | Simulate a specific attack type |
| `WS` | `/ws/threats` | Real-time threat event stream |

### Example: Simulate an Attack

```bash
curl -X POST http://localhost:8000/api/simulate/attack \
  -H "Content-Type: application/json" \
  -d '{"attack_type": "ddos"}'
```

---

## Project Structure

```
cyber_ai/
├── main.py                 ← FastAPI server entry point
├── requirements.txt        ← Python dependencies
├── models/
│   ├── train.py            ← Model training pipeline
│   ├── predict.py          ← Prediction engine
│   └── saved/              ← Serialized .pkl model files
├── data/
│   ├── generate_data.py    ← Synthetic data generator
│   └── sample_traffic.csv  ← Generated training data
├── static/
│   └── integration.js      ← Frontend ↔ Backend bridge
└── README.md
```

---

## ML Models

| Model | Algorithm | Purpose | Key Params |
|-------|-----------|---------|------------|
| Anomaly Detector | Isolation Forest | Flag unknown/zero-day threats | contamination=0.1, n_estimators=200 |
| Threat Classifier | Random Forest | Classify attack types | n_estimators=200, max_depth=15 |
| Behavior Clusterer | KMeans | Group traffic patterns | n_clusters=6 |

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **ML**: scikit-learn (Isolation Forest, Random Forest, KMeans)
- **Data**: pandas, numpy, joblib
- **Frontend**: HTML/CSS/JS, D3.js, WebSocket
- **Protocol**: REST API + WebSocket for real-time streaming

---

## License

Built for hackathon demonstration purposes.
