"""
Prediction Engine for NeuroSent Cyber Threat Intelligence.

Loads trained ML models and runs real-time inference on network traffic data.
Returns structured threat intelligence with SHAP-style feature attribution.
"""

import os
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import joblib


FEATURE_COLUMNS = [
    "duration", "protocol_type", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_serror_rate",
]

COUNTRY_DATA = [
    {"country": "China", "flag": "🇨🇳", "code": "CN"},
    {"country": "Russia", "flag": "🇷🇺", "code": "RU"},
    {"country": "United States", "flag": "🇺🇸", "code": "US"},
    {"country": "Brazil", "flag": "🇧🇷", "code": "BR"},
    {"country": "North Korea", "flag": "🇰🇵", "code": "KP"},
    {"country": "Iran", "flag": "🇮🇷", "code": "IR"},
    {"country": "Germany", "flag": "🇩🇪", "code": "DE"},
    {"country": "India", "flag": "🇮🇳", "code": "IN"},
    {"country": "Vietnam", "flag": "🇻🇳", "code": "VN"},
    {"country": "Indonesia", "flag": "🇮🇩", "code": "ID"},
]

THREAT_EMOJI = {
    "ddos": "⚠ DDoS Attack Detected",
    "port_scan": "◆ Port Scan Activity",
    "brute_force": "⚠ Brute Force Login",
    "sql_injection": "⚠ SQL Injection Attempt",
    "malware_c2": "☠ Malware C2 Communication",
    "zero_day": "🔴 Zero-Day Exploit",
    "normal": "✓ Normal Traffic",
}

# Country-specific IP ranges for realism
COUNTRY_IP_RANGES = {
    "China": ["218.92", "116.31", "61.135", "202.108", "180.101"],
    "Russia": ["77.88", "91.108", "5.188", "45.33", "185.220"],
    "United States": ["198.23", "162.158", "104.16", "8.8", "172.217"],
    "Brazil": ["200.147", "177.71", "189.90", "201.49", "187.72"],
    "North Korea": ["175.45", "210.52", "175.45", "210.52", "175.45"],
    "Iran": ["5.160", "91.98", "78.38", "2.144", "31.14"],
    "Germany": ["46.4", "138.201", "88.198", "78.46", "144.76"],
    "India": ["103.21", "14.139", "117.18", "49.36", "122.176"],
    "Vietnam": ["113.161", "14.160", "27.72", "42.112", "171.244"],
    "Indonesia": ["114.4", "36.68", "103.28", "180.244", "110.138"],
}


class PredictionEngine:
    """Loads ML models and provides real-time threat prediction."""

    def __init__(self):
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved")
        self.models_loaded = False

        try:
            self.scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
            self.label_encoder = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
            self.isolation_forest = joblib.load(os.path.join(base_dir, "isolation_forest.pkl"))
            self.random_forest = joblib.load(os.path.join(base_dir, "random_forest.pkl"))
            self.kmeans = joblib.load(os.path.join(base_dir, "kmeans.pkl"))
            self.feature_importances = self.random_forest.feature_importances_
            self.models_loaded = True
            print("[+] PredictionEngine: All models loaded successfully")
        except FileNotFoundError as e:
            print(f"[!] PredictionEngine: Model file not found — {e}")
            print("[!] Run 'python models/train.py' first to generate model files.")

    def _generate_ip(self, country: Optional[str] = None) -> str:
        """Generate a realistic-looking IP address from a country."""
        if country and country in COUNTRY_IP_RANGES:
            prefix = random.choice(COUNTRY_IP_RANGES[country])
            parts = prefix.split(".")
            while len(parts) < 4:
                parts.append(str(random.randint(1, 254)))
            return ".".join(parts[:4])
        return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"

    def _get_severity(self, confidence: float, is_anomaly: bool) -> str:
        """Compute threat severity from confidence and anomaly status."""
        if confidence > 0.9 and is_anomaly:
            return "CRITICAL"
        elif confidence > 0.75:
            return "HIGH"
        elif confidence > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _compute_shap_features(self, feature_values: np.ndarray) -> List[Dict]:
        """
        Compute SHAP-style feature importance for a single prediction.
        Uses RF feature importances weighted by the sample's scaled feature values.
        """
        abs_vals = np.abs(feature_values.flatten())
        weighted = abs_vals * self.feature_importances
        if weighted.max() > 0:
            weighted = weighted / weighted.max()

        top_indices = np.argsort(weighted)[::-1][:3]
        shap_features = []
        for idx in top_indices:
            shap_features.append({
                "feature": FEATURE_COLUMNS[idx],
                "value": round(float(weighted[idx]), 2),
            })
        return shap_features

    def predict(self, traffic_data: Dict) -> Dict:
        """
        Run full prediction pipeline on a single traffic sample.

        Args:
            traffic_data: dict with feature column keys and numeric values.

        Returns:
            Structured threat intelligence dict.
        """
        if not self.models_loaded:
            return {"error": "Run python models/train.py first"}

        # Build feature vector
        features = []
        for col in FEATURE_COLUMNS:
            features.append(float(traffic_data.get(col, 0.0)))
        feature_array = np.array(features).reshape(1, -1)

        # Scale
        X_scaled = self.scaler.transform(feature_array)

        # Isolation Forest — anomaly detection
        anomaly_pred = self.isolation_forest.predict(X_scaled)[0]
        anomaly_score_raw = self.isolation_forest.decision_function(X_scaled)[0]
        is_anomaly = anomaly_pred == -1

        # Random Forest — classification
        rf_pred = self.random_forest.predict(X_scaled)[0]
        rf_proba = self.random_forest.predict_proba(X_scaled)[0]
        confidence = float(np.max(rf_proba))
        threat_label = self.label_encoder.inverse_transform([rf_pred])[0]

        # KMeans — clustering
        cluster_id = int(self.kmeans.predict(X_scaled)[0])

        # SHAP-style feature attribution
        shap_features = self._compute_shap_features(X_scaled)

        # Determine threat status
        threat_detected = threat_label != "normal"

        # Severity
        severity = self._get_severity(confidence, is_anomaly)
        if not threat_detected:
            severity = "LOW"

        # Generate realistic IPs and country
        country_info = random.choice(COUNTRY_DATA)
        country = traffic_data.get("_country", country_info["country"])
        source_ip = traffic_data.get("_source_ip", self._generate_ip(country))
        dest_ip = traffic_data.get("_dest_ip", self._generate_ip("United States"))

        # Timestamp
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")

        return {
            "threat_detected": threat_detected,
            "threat_type": threat_label,
            "severity": severity,
            "confidence": round(confidence, 4),
            "anomaly_score": round(float(anomaly_score_raw), 4),
            "cluster_id": cluster_id,
            "source_ip": source_ip,
            "destination_ip": dest_ip,
            "country": country,
            "shap_features": shap_features,
            "timestamp": timestamp,
            "raw_features": {col: round(float(traffic_data.get(col, 0)), 4) for col in FEATURE_COLUMNS},
        }

    def predict_batch(self, traffic_list: List[Dict]) -> List[Dict]:
        """Run predictions on a list of traffic samples."""
        return [self.predict(t) for t in traffic_list]

    def generate_random_traffic(self, attack_type: Optional[str] = None) -> Dict:
        """
        Generate a single random traffic sample, optionally of a specific attack type.
        Returns a feature dict suitable for self.predict().
        """
        if attack_type == "ddos":
            return self._gen_ddos_sample()
        elif attack_type == "port_scan":
            return self._gen_port_scan_sample()
        elif attack_type == "brute_force":
            return self._gen_brute_force_sample()
        elif attack_type == "sql_injection":
            return self._gen_sql_injection_sample()
        elif attack_type == "malware_c2":
            return self._gen_malware_c2_sample()
        elif attack_type == "zero_day":
            return self._gen_zero_day_sample()
        else:
            # Random mix: 30% normal, 70% attacks for more interesting demo
            if random.random() < 0.3:
                return self._gen_normal_sample()
            else:
                attack = random.choice(["ddos", "port_scan", "brute_force",
                                        "sql_injection", "malware_c2", "zero_day"])
                return self.generate_random_traffic(attack)

    def _gen_normal_sample(self) -> Dict:
        return {
            "duration": round(random.expovariate(1 / 50), 2),
            "protocol_type": random.choice([0, 1, 2]),
            "src_bytes": round(random.lognormvariate(6, 1.5), 0),
            "dst_bytes": round(random.lognormvariate(7, 1.2), 0),
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": random.randint(0, 2),
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": random.randint(0, 1),
            "num_shells": 0,
            "num_access_files": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": random.randint(1, 40),
            "srv_count": random.randint(1, 25),
            "serror_rate": round(random.uniform(0, 0.05), 4),
            "rerror_rate": round(random.uniform(0, 0.05), 4),
            "same_srv_rate": round(random.uniform(0.8, 1.0), 4),
            "diff_srv_rate": round(random.uniform(0.0, 0.15), 4),
            "dst_host_count": random.randint(80, 255),
            "dst_host_srv_count": random.randint(40, 255),
            "dst_host_same_srv_rate": round(random.uniform(0.7, 1.0), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.0, 0.15), 4),
            "dst_host_serror_rate": round(random.uniform(0, 0.03), 4),
        }

    def _gen_ddos_sample(self) -> Dict:
        country = random.choice(["China", "Russia", "North Korea"])
        return {
            "duration": round(random.uniform(0, 2), 2),
            "protocol_type": random.choice([0, 1]),
            "src_bytes": round(random.lognormvariate(10, 0.8), 0),
            "dst_bytes": round(random.uniform(0, 300), 0),
            "land": random.choice([0, 0, 0, 0, 1]),
            "wrong_fragment": random.choice([0, 0, 1, 2]),
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": random.randint(250, 511),
            "srv_count": random.randint(1, 8),
            "serror_rate": round(random.uniform(0.85, 1.0), 4),
            "rerror_rate": round(random.uniform(0.0, 0.15), 4),
            "same_srv_rate": round(random.uniform(0.0, 0.2), 4),
            "diff_srv_rate": round(random.uniform(0.7, 1.0), 4),
            "dst_host_count": random.randint(220, 255),
            "dst_host_srv_count": random.randint(1, 20),
            "dst_host_same_srv_rate": round(random.uniform(0.0, 0.15), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.6, 1.0), 4),
            "dst_host_serror_rate": round(random.uniform(0.75, 1.0), 4),
            "_country": country,
            "_source_ip": self._generate_ip(country),
        }

    def _gen_port_scan_sample(self) -> Dict:
        country = random.choice(["Russia", "China", "Vietnam", "Indonesia"])
        return {
            "duration": round(random.uniform(0, 1.5), 2),
            "protocol_type": random.choice([0, 1]),
            "src_bytes": round(random.uniform(0, 150), 0),
            "dst_bytes": round(random.uniform(0, 80), 0),
            "land": 0,
            "wrong_fragment": random.choice([0, 0, 1]),
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": random.randint(150, 511),
            "srv_count": random.randint(150, 511),
            "serror_rate": round(random.uniform(0.0, 0.25), 4),
            "rerror_rate": round(random.uniform(0.65, 1.0), 4),
            "same_srv_rate": round(random.uniform(0.0, 0.1), 4),
            "diff_srv_rate": round(random.uniform(0.85, 1.0), 4),
            "dst_host_count": random.randint(220, 255),
            "dst_host_srv_count": random.randint(1, 15),
            "dst_host_same_srv_rate": round(random.uniform(0.0, 0.08), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.85, 1.0), 4),
            "dst_host_serror_rate": round(random.uniform(0.0, 0.15), 4),
            "_country": country,
            "_source_ip": self._generate_ip(country),
        }

    def _gen_brute_force_sample(self) -> Dict:
        country = random.choice(["Russia", "China", "Iran", "Brazil"])
        return {
            "duration": round(random.uniform(2, 25), 2),
            "protocol_type": random.choice([0, 1, 1, 1]),
            "src_bytes": round(random.uniform(200, 1800), 0),
            "dst_bytes": round(random.uniform(600, 4500), 0),
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": random.randint(1, 8),
            "num_failed_logins": random.randint(5, 18),
            "logged_in": random.choice([0, 0, 0, 1]),
            "num_compromised": random.choice([0, 0, 1]),
            "root_shell": random.choice([0, 0, 0, 1]),
            "su_attempted": random.choice([0, 0, 1]),
            "num_root": random.randint(0, 3),
            "num_file_creations": 0,
            "num_shells": random.choice([0, 0, 1]),
            "num_access_files": random.randint(0, 2),
            "is_host_login": random.choice([0, 0, 1]),
            "is_guest_login": random.choice([0, 0, 1]),
            "count": random.randint(60, 180),
            "srv_count": random.randint(8, 40),
            "serror_rate": round(random.uniform(0.0, 0.2), 4),
            "rerror_rate": round(random.uniform(0.35, 0.75), 4),
            "same_srv_rate": round(random.uniform(0.75, 1.0), 4),
            "diff_srv_rate": round(random.uniform(0.0, 0.25), 4),
            "dst_host_count": random.randint(1, 25),
            "dst_host_srv_count": random.randint(1, 15),
            "dst_host_same_srv_rate": round(random.uniform(0.85, 1.0), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.0, 0.15), 4),
            "dst_host_serror_rate": round(random.uniform(0.0, 0.1), 4),
            "_country": country,
            "_source_ip": self._generate_ip(country),
        }

    def _gen_sql_injection_sample(self) -> Dict:
        country = random.choice(["China", "Brazil", "India", "Russia"])
        return {
            "duration": round(random.uniform(2, 50), 2),
            "protocol_type": random.choice([1, 1, 1, 0]),
            "src_bytes": round(random.lognormvariate(8, 0.8), 0),
            "dst_bytes": round(random.lognormvariate(9, 1.2), 0),
            "land": 0,
            "wrong_fragment": 0,
            "urgent": random.choice([0, 0, 1]),
            "hot": random.randint(4, 14),
            "num_failed_logins": random.choice([0, 0, 1]),
            "logged_in": random.choice([0, 1, 1]),
            "num_compromised": random.randint(1, 8),
            "root_shell": random.choice([0, 0, 1]),
            "su_attempted": random.choice([0, 0, 1]),
            "num_root": random.randint(0, 6),
            "num_file_creations": random.randint(1, 6),
            "num_shells": random.choice([0, 0, 1]),
            "num_access_files": random.randint(1, 5),
            "is_host_login": 0,
            "is_guest_login": random.choice([0, 0, 1]),
            "count": random.randint(1, 25),
            "srv_count": random.randint(1, 15),
            "serror_rate": round(random.uniform(0.0, 0.15), 4),
            "rerror_rate": round(random.uniform(0.0, 0.15), 4),
            "same_srv_rate": round(random.uniform(0.85, 1.0), 4),
            "diff_srv_rate": round(random.uniform(0.0, 0.15), 4),
            "dst_host_count": random.randint(1, 40),
            "dst_host_srv_count": random.randint(1, 40),
            "dst_host_same_srv_rate": round(random.uniform(0.75, 1.0), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.0, 0.25), 4),
            "dst_host_serror_rate": round(random.uniform(0.0, 0.08), 4),
            "_country": country,
            "_source_ip": self._generate_ip(country),
        }

    def _gen_malware_c2_sample(self) -> Dict:
        country = random.choice(["North Korea", "Russia", "Iran", "China"])
        return {
            "duration": random.choice([30, 60, 120, 300, 600]) + random.gauss(0, 3),
            "protocol_type": random.choice([0, 1, 2]),
            "src_bytes": round(random.uniform(60, 450), 0),
            "dst_bytes": round(random.lognormvariate(7, 1.8), 0),
            "land": 0,
            "wrong_fragment": random.choice([0, 0, 0, 1]),
            "urgent": 0,
            "hot": random.randint(0, 4),
            "num_failed_logins": 0,
            "logged_in": 1,
            "num_compromised": random.randint(2, 12),
            "root_shell": random.choice([0, 1]),
            "su_attempted": random.choice([0, 0, 1]),
            "num_root": random.randint(1, 8),
            "num_file_creations": random.randint(2, 10),
            "num_shells": random.choice([0, 1, 2]),
            "num_access_files": random.randint(2, 7),
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": random.randint(1, 12),
            "srv_count": random.randint(1, 8),
            "serror_rate": round(random.uniform(0.0, 0.08), 4),
            "rerror_rate": round(random.uniform(0.0, 0.08), 4),
            "same_srv_rate": round(random.uniform(0.92, 1.0), 4),
            "diff_srv_rate": round(random.uniform(0.0, 0.08), 4),
            "dst_host_count": random.randint(1, 8),
            "dst_host_srv_count": random.randint(1, 8),
            "dst_host_same_srv_rate": round(random.uniform(0.92, 1.0), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.0, 0.08), 4),
            "dst_host_serror_rate": round(random.uniform(0.0, 0.04), 4),
            "_country": country,
            "_source_ip": self._generate_ip(country),
        }

    def _gen_zero_day_sample(self) -> Dict:
        country = random.choice(["North Korea", "China", "Iran", "Russia"])
        return {
            "duration": round(random.expovariate(1 / 100), 2),
            "protocol_type": random.choice([0, 1, 2]),
            "src_bytes": round(random.lognormvariate(9, 1.8), 0),
            "dst_bytes": round(random.lognormvariate(8, 2.2), 0),
            "land": random.choice([0, 0, 1]),
            "wrong_fragment": random.choice([0, 1, 2]),
            "urgent": random.choice([0, 0, 1]),
            "hot": random.randint(6, 22),
            "num_failed_logins": random.randint(0, 4),
            "logged_in": random.choice([0, 1]),
            "num_compromised": random.randint(3, 18),
            "root_shell": random.choice([0, 1, 1]),
            "su_attempted": random.choice([0, 1, 2]),
            "num_root": random.randint(2, 12),
            "num_file_creations": random.randint(3, 18),
            "num_shells": random.choice([0, 1, 2, 3]),
            "num_access_files": random.randint(3, 9),
            "is_host_login": random.choice([0, 0, 1]),
            "is_guest_login": random.choice([0, 1]),
            "count": random.randint(1, 500),
            "srv_count": random.randint(1, 500),
            "serror_rate": round(random.uniform(0.0, 1.0), 4),
            "rerror_rate": round(random.uniform(0.0, 1.0), 4),
            "same_srv_rate": round(random.uniform(0.0, 1.0), 4),
            "diff_srv_rate": round(random.uniform(0.0, 1.0), 4),
            "dst_host_count": random.randint(1, 255),
            "dst_host_srv_count": random.randint(1, 255),
            "dst_host_same_srv_rate": round(random.uniform(0.0, 1.0), 4),
            "dst_host_diff_srv_rate": round(random.uniform(0.0, 1.0), 4),
            "dst_host_serror_rate": round(random.uniform(0.0, 1.0), 4),
            "_country": country,
            "_source_ip": self._generate_ip(country),
        }
