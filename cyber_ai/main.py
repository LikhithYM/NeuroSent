"""
NeuroSent — FastAPI Backend Server

Serves the dashboard, REST API, and WebSocket real-time threat feed.
All predictions use real ML models trained via models/train.py.
"""

import asyncio
import json
import os
import random
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models.predict import PredictionEngine, COUNTRY_DATA

# ─── Paths ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_CYBER = os.path.join(BASE_DIR, "index.html")
INDEX_HTML_ROOT = os.path.join(BASE_DIR, "..", "index.html")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ─── Global state ───
engine: Optional[PredictionEngine] = None
stats_state: Dict = {}
real_attack_queues: List[asyncio.Queue] = []
ip_tracker = defaultdict(lambda: {"req_count": 0, "404_count": 0, "login_count": 0, "last_reset": time.time()})


def _init_stats():
    """Initialize live stats state."""
    return {
        "threats_today": random.randint(2500, 3500),
        "active_alerts": random.randint(100, 200),
        "detection_rate": round(random.uniform(97.5, 99.5), 1),
        "false_positive_rate": round(random.uniform(0.1, 0.8), 1),
        "events_per_second": random.randint(3500, 6000),
        "attack_breakdown": {
            "ddos": 0,
            "port_scan": 0,
            "brute_force": 0,
            "sql_injection": 0,
            "malware_c2": 0,
            "zero_day": 0,
        },
        "top_source_countries": [
            {"country": "China", "flag": "🇨🇳", "percentage": 28},
            {"country": "Russia", "flag": "🇷🇺", "percentage": 21},
            {"country": "United States", "flag": "🇺🇸", "percentage": 11},
            {"country": "Brazil", "flag": "🇧🇷", "percentage": 9},
            {"country": "North Korea", "flag": "🇰🇵", "percentage": 7},
        ],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize prediction engine and stats on startup."""
    global engine, stats_state
    engine = PredictionEngine()
    stats_state = _init_stats()
    print("[*] NeuroSent server starting ...")
    yield
    print("[*] NeuroSent server shutting down ...")


app = FastAPI(
    title="NeuroSent — Cyber Threat Intelligence",
    description="AI-powered real-time cyber threat detection system",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files ───
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Request/Response models ───
class TrafficFeatures(BaseModel):
    duration: float = 0
    protocol_type: float = 0
    src_bytes: float = 0
    dst_bytes: float = 0
    land: float = 0
    wrong_fragment: float = 0
    urgent: float = 0
    hot: float = 0
    num_failed_logins: float = 0
    logged_in: float = 0
    num_compromised: float = 0
    root_shell: float = 0
    su_attempted: float = 0
    num_root: float = 0
    num_file_creations: float = 0
    num_shells: float = 0
    num_access_files: float = 0
    is_host_login: float = 0
    is_guest_login: float = 0
    count: float = 0
    srv_count: float = 0
    serror_rate: float = 0
    rerror_rate: float = 0
    same_srv_rate: float = 0
    diff_srv_rate: float = 0
    dst_host_count: float = 0
    dst_host_srv_count: float = 0
    dst_host_same_srv_rate: float = 0
    dst_host_diff_srv_rate: float = 0
    dst_host_serror_rate: float = 0


class AttackSimRequest(BaseModel):
    attack_type: str


# ═══════════════════════════════════════════════════════
# LIVE ATTACK INTERCEPTOR (HONEYPOT MIDDLEWARE)
# ═══════════════════════════════════════════════════════

@app.middleware("http")
async def intercept_real_attacks(request: Request, call_next):
    """Monitors real HTTP traffic to detect live Kali attacks."""
    path = request.url.path
    
    # Ignore internal API and websocket noise
    if path.startswith("/ws") or path.startswith("/api/stats") or path.startswith("/static"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "Unknown"
    tracker = ip_tracker[client_ip]
    
    # Reset tracking window every 4 seconds
    now = time.time()
    if now - tracker["last_reset"] > 4.0:
        tracker["req_count"] = 0
        tracker["404_count"] = 0
        tracker["login_count"] = 0
        tracker["last_reset"] = now
        
    tracker["req_count"] += 1
    if "login" in path.lower():
        tracker["login_count"] += 1
        
    response = await call_next(request)
    
    if response.status_code == 404:
        tracker["404_count"] += 1
        
    # Detect thresholds
    attack_type = None
    if tracker["req_count"] > 30:          # >30 reqs in 4s -> DDoS / Flood
        attack_type = "ddos"
    elif tracker["404_count"] > 10:        # >10 404s in 4s -> Dirb / Port Scan
        attack_type = "port_scan"
    elif tracker["login_count"] > 5:       # >5 logins in 4s -> Hydra Brute Force
        attack_type = "brute_force"
        
    # If attack detected, inject into ML pipeline using REAL IP
    if attack_type and engine and engine.models_loaded:
        traffic = engine.generate_random_traffic(attack_type)
        traffic["_source_ip"] = client_ip
        traffic["_country"] = "Attacker (Kali)"
        
        prediction = engine.predict(traffic)
        _update_stats(prediction)
        
        # Broadcast to dashboard
        for q in real_attack_queues:
            await q.put(prediction)
            
        # Reset counters so we don't spam indefinitely
        tracker["req_count"] = 0
        tracker["404_count"] = 0
        tracker["login_count"] = 0
        tracker["last_reset"] = time.time()

    return response


# ═══════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════

@app.post("/login")
@app.get("/login")
async def honeypot_login():
    """Honeypot endpoint for Hydra brute-force demonstrations."""
    await asyncio.sleep(0.1) # Simulate slight delay
    return JSONResponse(status_code=401, content={"error": "Invalid credentials"})

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the dashboard HTML with integration.js injected."""
    # Try cyber_ai local copy first, then parent directory
    html_path = INDEX_HTML_CYBER if os.path.exists(INDEX_HTML_CYBER) else INDEX_HTML_ROOT
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Inject integration.js before </body> if not already present
    marker = "<!-- BACKEND_INTEGRATED -->"
    if marker not in html_content:
        injection = f'\n{marker}\n<script src="/static/integration.js"></script>\n'
        html_content = html_content.replace("</body>", injection + "</body>")

    return HTMLResponse(content=html_content)


@app.post("/api/predict")
async def predict_single(features: TrafficFeatures):
    """Run ML prediction on a single traffic sample."""
    if engine is None or not engine.models_loaded:
        raise HTTPException(status_code=503, detail="Run python models/train.py first")

    result = engine.predict(features.model_dump())
    _update_stats(result)
    return JSONResponse(content=result)


@app.post("/api/predict/batch")
async def predict_batch(features_list: List[TrafficFeatures]):
    """Run ML prediction on a batch of traffic samples."""
    if engine is None or not engine.models_loaded:
        raise HTTPException(status_code=503, detail="Run python models/train.py first")

    results = []
    for features in features_list:
        result = engine.predict(features.model_dump())
        _update_stats(result)
        results.append(result)
    return JSONResponse(content=results)


@app.get("/api/stats")
async def get_stats():
    """Return current dashboard statistics."""
    return JSONResponse(content=stats_state)


@app.post("/api/simulate/attack")
async def simulate_attack(request: AttackSimRequest):
    """Simulate a specific attack type and return the ML prediction."""
    if engine is None or not engine.models_loaded:
        raise HTTPException(status_code=503, detail="Run python models/train.py first")

    valid_types = ["ddos", "port_scan", "brute_force", "sql_injection", "malware_c2", "zero_day"]
    if request.attack_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid attack_type. Must be one of: {valid_types}"
        )

    traffic = engine.generate_random_traffic(request.attack_type)
    result = engine.predict(traffic)
    _update_stats(result)
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════
# WEBSOCKET — Real-Time Threat Feed
# ═══════════════════════════════════════════════════════

@app.websocket("/ws/threats")
async def websocket_threats(websocket: WebSocket):
    """Stream real-time ML threat predictions over WebSocket."""
    await websocket.accept()
    
    # Create a dedicated queue for this dashboard client
    client_queue = asyncio.Queue()
    real_attack_queues.append(client_queue)
    print(f"[+] WebSocket client connected ({len(real_attack_queues)} total)")

    threat_counter = 0
    try:
        while True:
            try:
                # Wait for a real manual attack from Kali (up to 1.5 seconds)
                prediction = await asyncio.wait_for(client_queue.get(), timeout=1.5)
            except asyncio.TimeoutError:
                # If no real attack, ONLY generate normal background simulation traffic
                if engine and engine.models_loaded:
                    traffic = engine.generate_random_traffic(attack_type="normal")
                    prediction = engine.predict(traffic)
                    _update_stats(prediction)
                else:
                    prediction = None

            if prediction:
                # Send threat event
                await websocket.send_json({
                    "type": "threat",
                    "data": prediction,
                })

                threat_counter += 1

                # Send stats every ~4 threat events
                if threat_counter % 4 == 0:
                    # Jitter the stats for liveliness
                    stats_state["events_per_second"] = random.randint(3500, 6000)
                    stats_state["threats_today"] += random.randint(1, 5)
                    stats_state["active_alerts"] = max(50, stats_state["active_alerts"] + random.randint(-3, 5))
                    stats_state["detection_rate"] = round(random.uniform(97.5, 99.5), 1)
                    stats_state["false_positive_rate"] = round(random.uniform(0.1, 0.8), 1)

                    # Update country percentages with slight jitter
                    for c in stats_state["top_source_countries"]:
                        c["percentage"] = max(1, c["percentage"] + random.randint(-2, 2))

                    await websocket.send_json({
                        "type": "stats",
                        "data": stats_state,
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[!] WebSocket error: {e}")
    finally:
        if client_queue in real_attack_queues:
            real_attack_queues.remove(client_queue)
        print(f"[-] WebSocket client disconnected ({len(real_attack_queues)} remaining)")


# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def _update_stats(prediction: Dict):
    """Update global stats based on a new prediction."""
    if prediction.get("threat_detected"):
        threat_type = prediction.get("threat_type", "")
        if threat_type in stats_state.get("attack_breakdown", {}):
            stats_state["attack_breakdown"][threat_type] += 1


# ─── Entry point ───
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
