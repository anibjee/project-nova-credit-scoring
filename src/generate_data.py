import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible RNG
RNG = np.random.default_rng()

REGIONS = ["metro", "suburban", "rural"]
GENDERS = ["female", "male", "nonbinary"]
ROLES = ["driver", "merchant"]
VEHICLE_TYPES = ["car", "bike", "van"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n: int, seed: int = 42) -> pd.DataFrame:
    global RNG
    RNG = np.random.default_rng(seed)

    partner_id = np.arange(1, n + 1)

    role = RNG.choice(ROLES, size=n, p=[0.7, 0.3])
    region = RNG.choice(REGIONS, size=n, p=[0.5, 0.35, 0.15])
    gender = RNG.choice(GENDERS, size=n, p=[0.45, 0.5, 0.05])

    age = RNG.integers(18, 65, size=n)
    tenure_months = RNG.integers(1, 84, size=n)

    # Earnings patterns by role/region
    base_earn = np.where(role == "driver",
                         RNG.normal(900, 250, size=n),
                         RNG.normal(1200, 400, size=n))

    region_adj = np.select([
        region == "metro", region == "suburban", region == "rural"
    ], [1.15, 1.0, 0.85], default=1.0)

    earnings_monthly = np.clip(base_earn * region_adj, 200, None)

    trips_weekly = np.where(role == "driver",
                            RNG.normal(110, 30, size=n),
                            RNG.normal(65, 20, size=n))
    trips_weekly = np.clip(trips_weekly, 5, None)

    on_time_rate = np.clip(RNG.normal(0.93, 0.05, size=n), 0.5, 1.0)
    cancel_rate = np.clip(RNG.normal(0.04, 0.03, size=n), 0.0, 0.4)

    customer_rating = np.clip(RNG.normal(4.7, 0.25, size=n), 2.5, 5.0)
    safety_incidents_12m = RNG.poisson(0.05, size=n)

    # Merchant-like transactional patterns (also some drivers have wallet activity)
    wallet_txn_count_monthly = np.clip(RNG.normal(85, 35, size=n), 0, None)
    wallet_txn_value_monthly = np.clip(RNG.normal(450, 180, size=n), 0, None)

    income_volatility = np.clip(RNG.normal(0.25, 0.12, size=n), 0.0, 1.5)  # coeff of variation proxy
    seasonality_index = np.clip(RNG.normal(1.0, 0.25, size=n), 0.3, 2.0)

    prior_loans = RNG.binomial(1, 0.18, size=n)
    prior_defaults = RNG.binomial(1, 0.05 + 0.02 * prior_loans, size=n)

    # Driver-specific features
    vehicle_type = np.where(role == "driver",
                            RNG.choice(VEHICLE_TYPES, size=n, p=[0.6, 0.3, 0.1]),
                            "none")
    fuel_cost_share = np.where(role == "driver",
                               np.clip(RNG.normal(0.18, 0.05, size=n), 0.05, 0.5),
                               0.0)

    # Construct latent default risk (higher = riskier)
    # Use interpretable relationships: higher earnings, higher on-time, lower cancel, higher rating -> lower risk
    # Penalize extreme volatility, many safety incidents, very short tenure
    z = (
        -0.002 * (earnings_monthly - 1000)  # normalize around typical earnings
        -3.0 * (on_time_rate - 0.7)  # strong signal from reliability
        +8.0 * cancel_rate  # strong penalty for cancellations
        -1.5 * (customer_rating - 3.0)  # reward good ratings
        +2.0 * income_volatility  # penalize volatility
        +0.8 * (seasonality_index - 1.0)  # seasonality impact
        +1.0 * safety_incidents_12m  # safety incidents penalty
        -0.02 * (tenure_months - 12)  # reward tenure beyond 1 year
        -0.01 * (trips_weekly - 50)  # reward high activity
        +0.5 * fuel_cost_share  # higher costs = potential strain
        +2.0 * prior_defaults  # strong signal from past defaults
        +0.5 * prior_loans  # mild signal from credit seeking
    )

    # Mild spurious correlations by region/role to emulate potential bias in observed outcomes
    z += np.select([region == "rural", region == "suburban", region == "metro"],
                   [0.15, 0.05, -0.05], default=0.0)
    z += np.where(role == "driver", 0.0, -0.05)  # merchants slightly less risky on average

    # Convert latent risk to probability via logistic
    base_prob = sigmoid(z)

    # Calibrate to realistic default rate (~8-12%)
    # Scale and clip
    default_prob = np.clip(0.08 + 0.9 * (base_prob - base_prob.mean()), 0.01, 0.6)

    defaulted_12m = RNG.binomial(1, default_prob)

    df = pd.DataFrame({
        "partner_id": partner_id,
        "role": role,
        "region": region,
        "gender": gender,
        "age": age,
        "tenure_months": tenure_months,
        "earnings_monthly": earnings_monthly,
        "trips_weekly": trips_weekly,
        "on_time_rate": on_time_rate,
        "cancel_rate": cancel_rate,
        "customer_rating": customer_rating,
        "safety_incidents_12m": safety_incidents_12m,
        "wallet_txn_count_monthly": wallet_txn_count_monthly,
        "wallet_txn_value_monthly": wallet_txn_value_monthly,
        "income_volatility": income_volatility,
        "seasonality_index": seasonality_index,
        "prior_loans": prior_loans,
        "prior_defaults": prior_defaults,
        "vehicle_type": vehicle_type,
        "fuel_cost_share": fuel_cost_share,
        "defaulted_12m": defaulted_12m,
    })

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/partners.csv")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = simulate(args.n, args.seed)
    df.to_csv(out_path, index=False)

    meta = {
        "n_rows": int(args.n),
        "seed": int(args.seed),
        "label": "defaulted_12m",
        "sensitive_attributes": ["gender", "region", "role"],
    }
    (out_path.parent / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote dataset to {out_path} with shape {df.shape}")


if __name__ == "__main__":
    main()

