"""
Synthetic Network Traffic Data Generator for NeuroSent Cyber Threat Intelligence.

Generates 10,000 rows of realistic network traffic data across 7 classes:
  normal (60%), ddos, port_scan, brute_force, sql_injection, malware_c2, zero_day.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

FEATURE_COLUMNS = [
    "duration", "protocol_type", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_serror_rate",
]

TOTAL_ROWS = 10_000
NORMAL_RATIO = 0.60
ATTACK_TYPES = ["ddos", "port_scan", "brute_force", "sql_injection", "malware_c2", "zero_day"]


def _generate_normal(n: int) -> pd.DataFrame:
    """Generate normal (benign) traffic patterns."""
    data = {
        "duration": np.random.exponential(scale=50, size=n).clip(0, 500),
        "protocol_type": np.random.choice([0, 1, 2], size=n, p=[0.5, 0.35, 0.15]),
        "src_bytes": np.random.lognormal(mean=6, sigma=1.5, size=n).clip(0, 50000),
        "dst_bytes": np.random.lognormal(mean=7, sigma=1.2, size=n).clip(0, 60000),
        "land": np.zeros(n, dtype=int),
        "wrong_fragment": np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
        "urgent": np.zeros(n, dtype=int),
        "hot": np.random.poisson(lam=0.3, size=n).clip(0, 5),
        "num_failed_logins": np.random.choice([0, 1], size=n, p=[0.97, 0.03]),
        "logged_in": np.random.choice([0, 1], size=n, p=[0.3, 0.7]),
        "num_compromised": np.zeros(n, dtype=int),
        "root_shell": np.zeros(n, dtype=int),
        "su_attempted": np.zeros(n, dtype=int),
        "num_root": np.zeros(n, dtype=int),
        "num_file_creations": np.random.poisson(lam=0.2, size=n).clip(0, 3),
        "num_shells": np.zeros(n, dtype=int),
        "num_access_files": np.random.poisson(lam=0.1, size=n).clip(0, 2),
        "is_host_login": np.zeros(n, dtype=int),
        "is_guest_login": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "count": np.random.randint(1, 50, size=n),
        "srv_count": np.random.randint(1, 30, size=n),
        "serror_rate": np.random.uniform(0, 0.1, size=n),
        "rerror_rate": np.random.uniform(0, 0.1, size=n),
        "same_srv_rate": np.random.uniform(0.7, 1.0, size=n),
        "diff_srv_rate": np.random.uniform(0.0, 0.3, size=n),
        "dst_host_count": np.random.randint(50, 255, size=n),
        "dst_host_srv_count": np.random.randint(20, 255, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.6, 1.0, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.0, 0.2, size=n),
        "dst_host_serror_rate": np.random.uniform(0, 0.05, size=n),
        "label": ["normal"] * n,
    }
    return pd.DataFrame(data)


def _generate_ddos(n: int) -> pd.DataFrame:
    """Generate DDoS / flood traffic patterns — high volume, short duration."""
    data = {
        "duration": np.random.uniform(0, 3, size=n),
        "protocol_type": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "src_bytes": np.random.lognormal(mean=10, sigma=1.0, size=n).clip(5000, 500000),
        "dst_bytes": np.random.uniform(0, 500, size=n),
        "land": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "wrong_fragment": np.random.choice([0, 1, 2, 3], size=n, p=[0.7, 0.15, 0.1, 0.05]),
        "urgent": np.zeros(n, dtype=int),
        "hot": np.zeros(n, dtype=int),
        "num_failed_logins": np.zeros(n, dtype=int),
        "logged_in": np.zeros(n, dtype=int),
        "num_compromised": np.zeros(n, dtype=int),
        "root_shell": np.zeros(n, dtype=int),
        "su_attempted": np.zeros(n, dtype=int),
        "num_root": np.zeros(n, dtype=int),
        "num_file_creations": np.zeros(n, dtype=int),
        "num_shells": np.zeros(n, dtype=int),
        "num_access_files": np.zeros(n, dtype=int),
        "is_host_login": np.zeros(n, dtype=int),
        "is_guest_login": np.zeros(n, dtype=int),
        "count": np.random.randint(200, 511, size=n),
        "srv_count": np.random.randint(1, 10, size=n),
        "serror_rate": np.random.uniform(0.8, 1.0, size=n),
        "rerror_rate": np.random.uniform(0.0, 0.2, size=n),
        "same_srv_rate": np.random.uniform(0.0, 0.3, size=n),
        "diff_srv_rate": np.random.uniform(0.6, 1.0, size=n),
        "dst_host_count": np.random.randint(200, 255, size=n),
        "dst_host_srv_count": np.random.randint(1, 30, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.0, 0.2, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.5, 1.0, size=n),
        "dst_host_serror_rate": np.random.uniform(0.7, 1.0, size=n),
        "label": ["ddos"] * n,
    }
    return pd.DataFrame(data)


def _generate_port_scan(n: int) -> pd.DataFrame:
    """Generate port scan patterns — many unique destinations, short connections."""
    data = {
        "duration": np.random.uniform(0, 2, size=n),
        "protocol_type": np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]),
        "src_bytes": np.random.uniform(0, 200, size=n),
        "dst_bytes": np.random.uniform(0, 100, size=n),
        "land": np.zeros(n, dtype=int),
        "wrong_fragment": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "urgent": np.zeros(n, dtype=int),
        "hot": np.zeros(n, dtype=int),
        "num_failed_logins": np.zeros(n, dtype=int),
        "logged_in": np.zeros(n, dtype=int),
        "num_compromised": np.zeros(n, dtype=int),
        "root_shell": np.zeros(n, dtype=int),
        "su_attempted": np.zeros(n, dtype=int),
        "num_root": np.zeros(n, dtype=int),
        "num_file_creations": np.zeros(n, dtype=int),
        "num_shells": np.zeros(n, dtype=int),
        "num_access_files": np.zeros(n, dtype=int),
        "is_host_login": np.zeros(n, dtype=int),
        "is_guest_login": np.zeros(n, dtype=int),
        "count": np.random.randint(100, 511, size=n),
        "srv_count": np.random.randint(100, 511, size=n),
        "serror_rate": np.random.uniform(0.0, 0.3, size=n),
        "rerror_rate": np.random.uniform(0.6, 1.0, size=n),
        "same_srv_rate": np.random.uniform(0.0, 0.15, size=n),
        "diff_srv_rate": np.random.uniform(0.8, 1.0, size=n),
        "dst_host_count": np.random.randint(200, 255, size=n),
        "dst_host_srv_count": np.random.randint(1, 20, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.0, 0.1, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.8, 1.0, size=n),
        "dst_host_serror_rate": np.random.uniform(0.0, 0.2, size=n),
        "label": ["port_scan"] * n,
    }
    return pd.DataFrame(data)


def _generate_brute_force(n: int) -> pd.DataFrame:
    """Generate brute force login patterns — many failed logins."""
    data = {
        "duration": np.random.uniform(1, 30, size=n),
        "protocol_type": np.random.choice([0, 1, 2], size=n, p=[0.3, 0.6, 0.1]),
        "src_bytes": np.random.uniform(100, 2000, size=n),
        "dst_bytes": np.random.uniform(500, 5000, size=n),
        "land": np.zeros(n, dtype=int),
        "wrong_fragment": np.zeros(n, dtype=int),
        "urgent": np.zeros(n, dtype=int),
        "hot": np.random.poisson(lam=2, size=n).clip(0, 10),
        "num_failed_logins": np.random.randint(3, 20, size=n),
        "logged_in": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "num_compromised": np.random.choice([0, 1, 2], size=n, p=[0.6, 0.3, 0.1]),
        "root_shell": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "su_attempted": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "num_root": np.random.poisson(lam=0.5, size=n).clip(0, 5),
        "num_file_creations": np.zeros(n, dtype=int),
        "num_shells": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "num_access_files": np.random.poisson(lam=0.3, size=n).clip(0, 3),
        "is_host_login": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "is_guest_login": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "count": np.random.randint(50, 200, size=n),
        "srv_count": np.random.randint(5, 50, size=n),
        "serror_rate": np.random.uniform(0.0, 0.3, size=n),
        "rerror_rate": np.random.uniform(0.3, 0.8, size=n),
        "same_srv_rate": np.random.uniform(0.7, 1.0, size=n),
        "diff_srv_rate": np.random.uniform(0.0, 0.3, size=n),
        "dst_host_count": np.random.randint(1, 30, size=n),
        "dst_host_srv_count": np.random.randint(1, 20, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.8, 1.0, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.0, 0.2, size=n),
        "dst_host_serror_rate": np.random.uniform(0.0, 0.15, size=n),
        "label": ["brute_force"] * n,
    }
    return pd.DataFrame(data)


def _generate_sql_injection(n: int) -> pd.DataFrame:
    """Generate SQL injection patterns — higher src_bytes, hot indicators."""
    data = {
        "duration": np.random.uniform(1, 60, size=n),
        "protocol_type": np.random.choice([0, 1, 2], size=n, p=[0.2, 0.7, 0.1]),
        "src_bytes": np.random.lognormal(mean=8, sigma=1.0, size=n).clip(500, 100000),
        "dst_bytes": np.random.lognormal(mean=9, sigma=1.5, size=n).clip(1000, 200000),
        "land": np.zeros(n, dtype=int),
        "wrong_fragment": np.zeros(n, dtype=int),
        "urgent": np.random.choice([0, 1], size=n, p=[0.9, 0.1]),
        "hot": np.random.randint(3, 15, size=n),
        "num_failed_logins": np.random.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        "logged_in": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
        "num_compromised": np.random.poisson(lam=2, size=n).clip(0, 10),
        "root_shell": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "su_attempted": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "num_root": np.random.poisson(lam=1, size=n).clip(0, 8),
        "num_file_creations": np.random.poisson(lam=1.5, size=n).clip(0, 8),
        "num_shells": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        "num_access_files": np.random.poisson(lam=2, size=n).clip(0, 6),
        "is_host_login": np.zeros(n, dtype=int),
        "is_guest_login": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "count": np.random.randint(1, 30, size=n),
        "srv_count": np.random.randint(1, 20, size=n),
        "serror_rate": np.random.uniform(0.0, 0.2, size=n),
        "rerror_rate": np.random.uniform(0.0, 0.2, size=n),
        "same_srv_rate": np.random.uniform(0.8, 1.0, size=n),
        "diff_srv_rate": np.random.uniform(0.0, 0.2, size=n),
        "dst_host_count": np.random.randint(1, 50, size=n),
        "dst_host_srv_count": np.random.randint(1, 50, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.7, 1.0, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.0, 0.3, size=n),
        "dst_host_serror_rate": np.random.uniform(0.0, 0.1, size=n),
        "label": ["sql_injection"] * n,
    }
    return pd.DataFrame(data)


def _generate_malware_c2(n: int) -> pd.DataFrame:
    """Generate malware command-and-control patterns — periodic beaconing."""
    data = {
        "duration": np.random.choice([30, 60, 120, 300, 600], size=n) + np.random.normal(0, 5, size=n),
        "protocol_type": np.random.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2]),
        "src_bytes": np.random.uniform(50, 500, size=n),
        "dst_bytes": np.random.lognormal(mean=7, sigma=2.0, size=n).clip(100, 80000),
        "land": np.zeros(n, dtype=int),
        "wrong_fragment": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "urgent": np.zeros(n, dtype=int),
        "hot": np.random.poisson(lam=1, size=n).clip(0, 5),
        "num_failed_logins": np.zeros(n, dtype=int),
        "logged_in": np.ones(n, dtype=int),
        "num_compromised": np.random.poisson(lam=3, size=n).clip(0, 15),
        "root_shell": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
        "su_attempted": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "num_root": np.random.poisson(lam=2, size=n).clip(0, 10),
        "num_file_creations": np.random.poisson(lam=3, size=n).clip(0, 12),
        "num_shells": np.random.choice([0, 1, 2], size=n, p=[0.5, 0.35, 0.15]),
        "num_access_files": np.random.poisson(lam=3, size=n).clip(0, 8),
        "is_host_login": np.zeros(n, dtype=int),
        "is_guest_login": np.zeros(n, dtype=int),
        "count": np.random.randint(1, 15, size=n),
        "srv_count": np.random.randint(1, 10, size=n),
        "serror_rate": np.random.uniform(0.0, 0.1, size=n),
        "rerror_rate": np.random.uniform(0.0, 0.1, size=n),
        "same_srv_rate": np.random.uniform(0.9, 1.0, size=n),
        "diff_srv_rate": np.random.uniform(0.0, 0.1, size=n),
        "dst_host_count": np.random.randint(1, 10, size=n),
        "dst_host_srv_count": np.random.randint(1, 10, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.9, 1.0, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.0, 0.1, size=n),
        "dst_host_serror_rate": np.random.uniform(0.0, 0.05, size=n),
        "label": ["malware_c2"] * n,
    }
    df = pd.DataFrame(data)
    df["duration"] = df["duration"].clip(0, 700)
    return df


def _generate_zero_day(n: int) -> pd.DataFrame:
    """Generate zero-day exploit patterns — anomalous combinations."""
    data = {
        "duration": np.random.exponential(scale=100, size=n).clip(0, 800),
        "protocol_type": np.random.choice([0, 1, 2], size=n),
        "src_bytes": np.random.lognormal(mean=9, sigma=2.0, size=n).clip(100, 300000),
        "dst_bytes": np.random.lognormal(mean=8, sigma=2.5, size=n).clip(0, 250000),
        "land": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "wrong_fragment": np.random.choice([0, 1, 2, 3], size=n, p=[0.5, 0.2, 0.2, 0.1]),
        "urgent": np.random.choice([0, 1, 2], size=n, p=[0.7, 0.2, 0.1]),
        "hot": np.random.randint(5, 25, size=n),
        "num_failed_logins": np.random.poisson(lam=1, size=n).clip(0, 5),
        "logged_in": np.random.choice([0, 1], size=n),
        "num_compromised": np.random.poisson(lam=5, size=n).clip(0, 20),
        "root_shell": np.random.choice([0, 1], size=n, p=[0.4, 0.6]),
        "su_attempted": np.random.choice([0, 1, 2], size=n, p=[0.5, 0.3, 0.2]),
        "num_root": np.random.poisson(lam=3, size=n).clip(0, 15),
        "num_file_creations": np.random.poisson(lam=5, size=n).clip(0, 20),
        "num_shells": np.random.choice([0, 1, 2, 3], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "num_access_files": np.random.poisson(lam=4, size=n).clip(0, 10),
        "is_host_login": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
        "is_guest_login": np.random.choice([0, 1], size=n, p=[0.5, 0.5]),
        "count": np.random.randint(1, 511, size=n),
        "srv_count": np.random.randint(1, 511, size=n),
        "serror_rate": np.random.uniform(0.0, 1.0, size=n),
        "rerror_rate": np.random.uniform(0.0, 1.0, size=n),
        "same_srv_rate": np.random.uniform(0.0, 1.0, size=n),
        "diff_srv_rate": np.random.uniform(0.0, 1.0, size=n),
        "dst_host_count": np.random.randint(1, 255, size=n),
        "dst_host_srv_count": np.random.randint(1, 255, size=n),
        "dst_host_same_srv_rate": np.random.uniform(0.0, 1.0, size=n),
        "dst_host_diff_srv_rate": np.random.uniform(0.0, 1.0, size=n),
        "dst_host_serror_rate": np.random.uniform(0.0, 1.0, size=n),
        "label": ["zero_day"] * n,
    }
    return pd.DataFrame(data)


GENERATORS = {
    "ddos": _generate_ddos,
    "port_scan": _generate_port_scan,
    "brute_force": _generate_brute_force,
    "sql_injection": _generate_sql_injection,
    "malware_c2": _generate_malware_c2,
    "zero_day": _generate_zero_day,
}


def generate_dataset() -> pd.DataFrame:
    """Generate the full synthetic dataset and save to CSV."""
    n_normal = int(TOTAL_ROWS * NORMAL_RATIO)
    n_attack = TOTAL_ROWS - n_normal
    n_per_attack = n_attack // len(ATTACK_TYPES)
    remainder = n_attack - n_per_attack * len(ATTACK_TYPES)

    frames = [_generate_normal(n_normal)]

    for i, attack in enumerate(ATTACK_TYPES):
        count = n_per_attack + (1 if i < remainder else 0)
        frames.append(GENERATORS[attack](count))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure correct column order
    df = df[FEATURE_COLUMNS + ["label"]]

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "sample_traffic.csv")
    df.to_csv(out_path, index=False)

    print(f"[+] Generated {len(df)} rows of synthetic traffic data")
    print(f"[+] Label distribution:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:20s}: {count:5d} ({pct:.1f}%)")
    print(f"[+] Saved to: {out_path}")

    return df


if __name__ == "__main__":
    generate_dataset()
