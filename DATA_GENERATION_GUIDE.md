# Data Generation Guide - Project Nova

This guide provides comprehensive instructions for generating different types of synthetic credit scoring data for the Project Nova system.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Basic Data Generation](#basic-data-generation)
3. [Advanced Data Variations](#advanced-data-variations)
4. [Specialized Datasets](#specialized-datasets)
5. [Time-Series Data](#time-series-data)
6. [Custom Data Modifications](#custom-data-modifications)
7. [Data Quality Controls](#data-quality-controls)
8. [External Data Integration](#external-data-integration)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Generate Standard Dataset
```bash
python src/generate_data.py --n 50000 --seed 42 --out data/partners.csv
```

### Generate Multiple Datasets
```bash
# Small test dataset
python src/generate_data.py --n 1000 --seed 100 --out data/test_partners.csv

# Large production dataset
python src/generate_data.py --n 100000 --seed 200 --out data/prod_partners.csv

# Validation dataset
python src/generate_data.py --n 25000 --seed 300 --out data/val_partners.csv
```

## Basic Data Generation

### Command Line Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--n` | Number of records | 50000 | `--n 10000` |
| `--seed` | Random seed for reproducibility | 42 | `--seed 123` |
| `--out` | Output file path | `data/partners.csv` | `--out data/custom.csv` |

### Standard Configurations

```bash
# Development (small, fast)
python src/generate_data.py --n 5000 --seed 42 --out data/dev_partners.csv

# Testing (medium size)
python src/generate_data.py --n 20000 --seed 42 --out data/test_partners.csv

# Production (large scale)
python src/generate_data.py --n 200000 --seed 42 --out data/prod_partners.csv

# Research (reproducible)
python src/generate_data.py --n 50000 --seed 12345 --out data/research_partners.csv
```

## Advanced Data Variations

### 1. Demographic Variations

To create datasets with different demographic distributions, modify `src/generate_data.py`:

#### More Driver-Heavy Dataset
```python
# Line 27: Change role distribution
role = RNG.choice(ROLES, size=n, p=[0.85, 0.15])  # 85% drivers, 15% merchants
```

#### Urban-Focused Dataset
```python
# Line 28: Change region distribution
region = RNG.choice(REGIONS, size=n, p=[0.70, 0.25, 0.05])  # 70% metro, 25% suburban, 5% rural
```

#### Gender-Balanced Dataset
```python
# Line 29: Change gender distribution
gender = RNG.choice(GENDERS, size=n, p=[0.48, 0.48, 0.04])  # Nearly equal male/female
```

### 2. Economic Variations

#### High-Earning Population
```python
# Lines 35-37: Modify base earnings
base_earn = np.where(role == "driver",
                     RNG.normal(1200, 300, size=n),  # Higher driver earnings
                     RNG.normal(1800, 500, size=n))  # Higher merchant earnings
```

#### Economic Recession Scenario
```python
# Lines 35-37: Lower earnings, higher volatility
base_earn = np.where(role == "driver",
                     RNG.normal(600, 200, size=n),   # Lower earnings
                     RNG.normal(800, 300, size=n))
# Line 60: Increase volatility
income_volatility = np.clip(RNG.normal(0.35, 0.15, size=n), 0.0, 2.0)
```

### 3. Risk Profile Variations

#### Low-Risk Population
```python
# Modify risk factors (lines 50-64)
on_time_rate = np.clip(RNG.normal(0.96, 0.03, size=n), 0.8, 1.0)  # Higher reliability
cancel_rate = np.clip(RNG.normal(0.02, 0.02, size=n), 0.0, 0.2)   # Lower cancellation
customer_rating = np.clip(RNG.normal(4.8, 0.15, size=n), 4.0, 5.0) # Higher ratings
prior_defaults = RNG.binomial(1, 0.02, size=n)  # Lower default history
```

#### High-Risk Population
```python
# Modify risk factors for higher risk
on_time_rate = np.clip(RNG.normal(0.85, 0.08, size=n), 0.5, 1.0)
cancel_rate = np.clip(RNG.normal(0.08, 0.05, size=n), 0.0, 0.5)
customer_rating = np.clip(RNG.normal(4.3, 0.35, size=n), 2.5, 5.0)
prior_defaults = RNG.binomial(1, 0.12, size=n)
```

## Specialized Datasets

### 1. Driver-Only Dataset

Create a specialized script `src/generate_driver_data.py`:

```python
def simulate_drivers_only(n: int, seed: int = 42) -> pd.DataFrame:
    # Force all records to be drivers
    role = np.full(n, "driver")
    
    # Driver-specific enhancements
    vehicle_age_months = RNG.integers(6, 120, size=n)
    maintenance_cost_monthly = RNG.normal(150, 50, size=n)
    
    # Continue with modified logic...
```

### 2. Merchant-Only Dataset

Create `src/generate_merchant_data.py`:

```python
def simulate_merchants_only(n: int, seed: int = 42) -> pd.DataFrame:
    # Force all records to be merchants
    role = np.full(n, "merchant")
    
    # Merchant-specific features
    business_type = RNG.choice(["restaurant", "retail", "service"], size=n)
    daily_transactions = RNG.poisson(45, size=n)
    peak_hours_ratio = RNG.normal(0.6, 0.15, size=n)
    
    # Continue with merchant-focused logic...
```

### 3. New Partner Dataset (First 6 Months)

```python
def simulate_new_partners(n: int, seed: int = 42) -> pd.DataFrame:
    # Limit tenure to first 6 months
    tenure_months = RNG.integers(1, 7, size=n)
    
    # Adjust other features for new partners
    trips_weekly = trips_weekly * 0.7  # Lower activity initially
    customer_rating = np.clip(RNG.normal(4.5, 0.3, size=n), 3.0, 5.0)  # More variation
```

## Time-Series Data

### 1. Monthly Data Generation

Create `src/generate_timeseries_data.py`:

```python
def generate_monthly_data(start_date="2022-01", end_date="2024-12", base_n=10000):
    """Generate monthly snapshots of partner data"""
    
    dates = pd.date_range(start_date, end_date, freq='MS')
    all_data = []
    
    for i, date in enumerate(dates):
        # Vary parameters by month
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
        n_partners = int(base_n * seasonal_factor)
        
        monthly_data = simulate(n_partners, seed=42 + i)
        monthly_data['data_month'] = date.strftime('%Y-%m')
        monthly_data['seasonal_factor'] = seasonal_factor
        
        all_data.append(monthly_data)
    
    return pd.concat(all_data, ignore_index=True)
```

### 2. Generate Historical Cohorts

```bash
# Generate data for different time periods
python src/generate_data.py --n 40000 --seed 2022 --out data/partners_2022.csv
python src/generate_data.py --n 45000 --seed 2023 --out data/partners_2023.csv  
python src/generate_data.py --n 50000 --seed 2024 --out data/partners_2024.csv
```

## Custom Data Modifications

### 1. Add New Features

Extend the data generation with additional features:

```python
# Add new features to the DataFrame (around line 106)
df["credit_inquiries_6m"] = RNG.poisson(1.2, size=n)
df["bank_account_age_months"] = RNG.integers(12, 240, size=n)
df["digital_wallet_usage"] = RNG.choice([0, 1], size=n, p=[0.3, 0.7])
df["social_media_verification"] = RNG.choice([0, 1], size=n, p=[0.6, 0.4])
df["referral_score"] = RNG.normal(0.5, 0.2, size=n)
```

### 2. Add Geographic Granularity

```python
# Replace simple regions with cities
CITIES = {
    "metro": ["Jakarta", "Manila", "Bangkok", "Kuala Lumpur", "Singapore"],
    "suburban": ["Bandung", "Cebu", "Chiang Mai", "Penang", "Johor"],
    "rural": ["Yogyakarta", "Davao", "Phuket", "Ipoh", "Melaka"]
}

def assign_cities(region_array):
    cities = []
    for region in region_array:
        cities.append(RNG.choice(CITIES[region]))
    return cities

# Add to DataFrame
df["city"] = assign_cities(df["region"].values)
```

### 3. Missing Data Simulation

```python
def introduce_missing_data(df, missing_rate=0.05):
    """Introduce realistic missing data patterns"""
    df_missing = df.copy()
    
    # Missing customer ratings (newer partners)
    new_partners = df_missing["tenure_months"] < 3
    missing_rating_idx = df_missing[new_partners].sample(frac=0.3).index
    df_missing.loc[missing_rating_idx, "customer_rating"] = np.nan
    
    # Missing wallet data (not all partners use wallet)
    missing_wallet_idx = df_missing.sample(frac=missing_rate).index
    df_missing.loc[missing_wallet_idx, ["wallet_txn_count_monthly", "wallet_txn_value_monthly"]] = np.nan
    
    return df_missing
```

## Data Quality Controls

### 1. Validation Checks

Create `src/validate_data.py`:

```python
def validate_generated_data(df):
    """Comprehensive data validation"""
    
    issues = []
    
    # Check data ranges
    if (df["age"] < 18).any() or (df["age"] > 65).any():
        issues.append("Age values outside expected range")
    
    if (df["customer_rating"] < 1.0).any() or (df["customer_rating"] > 5.0).any():
        issues.append("Customer rating outside valid range")
        
    if (df["on_time_rate"] < 0.0).any() or (df["on_time_rate"] > 1.0).any():
        issues.append("On-time rate outside valid range")
    
    # Check for realistic distributions
    default_rate = df["defaulted_12m"].mean()
    if default_rate < 0.05 or default_rate > 0.20:
        issues.append(f"Default rate ({default_rate:.3f}) outside realistic range")
    
    # Check missing values
    if df.isnull().sum().sum() > 0:
        issues.append("Unexpected missing values found")
    
    return issues
```

### 2. Statistical Consistency

```python
def check_statistical_consistency(df):
    """Check for statistical relationships"""
    
    # Higher earnings should correlate with lower default rates
    earnings_default_corr = df["earnings_monthly"].corr(df["defaulted_12m"])
    if earnings_default_corr > -0.1:
        print("Warning: Earnings-default correlation weaker than expected")
    
    # On-time rate should negatively correlate with defaults
    ontime_default_corr = df["on_time_rate"].corr(df["defaulted_12m"])
    if ontime_default_corr > -0.1:
        print("Warning: On-time rate not strongly protective against defaults")
```

## External Data Integration

### 1. Real Economic Indicators

```python
def apply_economic_context(df, unemployment_rate=0.05, inflation_rate=0.03):
    """Adjust synthetic data based on real economic conditions"""
    
    # Adjust earnings based on economic conditions
    economic_stress = unemployment_rate * 2 + inflation_rate
    earnings_adjustment = 1 - (economic_stress * 0.1)
    
    df["earnings_monthly"] *= earnings_adjustment
    df["income_volatility"] *= (1 + economic_stress)
    
    return df
```

### 2. Census Data Integration

```python
def apply_demographic_weights(df, census_data_path=None):
    """Reweight data to match real demographic distributions"""
    
    if census_data_path:
        census_df = pd.read_csv(census_data_path)
        # Apply reweighting logic based on census data
        # Implementation depends on census data format
    
    return df
```

## Batch Generation Scripts

### 1. Generate Multiple Scenarios

Create `scripts/generate_all_scenarios.py`:

```python
scenarios = {
    "baseline": {"n": 50000, "seed": 42},
    "high_risk": {"n": 30000, "seed": 100},
    "low_risk": {"n": 30000, "seed": 200},
    "driver_heavy": {"n": 40000, "seed": 300},
    "merchant_heavy": {"n": 40000, "seed": 400},
    "urban_focus": {"n": 35000, "seed": 500},
    "rural_focus": {"n": 25000, "seed": 600}
}

for scenario_name, params in scenarios.items():
    output_path = f"data/{scenario_name}_partners.csv"
    print(f"Generating {scenario_name}...")
    
    df = simulate(params["n"], params["seed"])
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
```

### 2. PowerShell Batch Script

Create `scripts/generate_datasets.ps1`:

```powershell
# Generate multiple datasets for different purposes

Write-Host "Generating Project Nova datasets..." -ForegroundColor Green

# Create data directory if it doesn't exist
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data"
}

# Development datasets
python src/generate_data.py --n 5000 --seed 42 --out data/dev_small.csv
python src/generate_data.py --n 15000 --seed 43 --out data/dev_medium.csv

# Testing datasets
python src/generate_data.py --n 25000 --seed 100 --out data/test_partners.csv
python src/generate_data.py --n 25000 --seed 101 --out data/test_partners_alt.csv

# Production datasets
python src/generate_data.py --n 100000 --seed 1000 --out data/prod_partners_v1.csv
python src/generate_data.py --n 120000 --seed 1001 --out data/prod_partners_v2.csv

# Specialized datasets
python src/generate_data.py --n 30000 --seed 2000 --out data/validation_set.csv
python src/generate_data.py --n 10000 --seed 3000 --out data/holdout_set.csv

Write-Host "Dataset generation complete!" -ForegroundColor Green
Write-Host "Check the 'data/' directory for generated files." -ForegroundColor Yellow
```

## Usage Examples

### 1. A/B Testing Data

```bash
# Control group
python src/generate_data.py --n 25000 --seed 500 --out data/control_group.csv

# Test group (with modifications)
python src/generate_data.py --n 25000 --seed 501 --out data/test_group.csv
```

### 2. Model Training Pipeline

```bash
# Training set (70%)
python src/generate_data.py --n 35000 --seed 42 --out data/train.csv

# Validation set (15%)
python src/generate_data.py --n 7500 --seed 43 --out data/val.csv

# Test set (15%)
python src/generate_data.py --n 7500 --seed 44 --out data/test.csv
```

### 3. Fairness Analysis

```bash
# Balanced demographics for fairness testing
python src/generate_data.py --n 40000 --seed 1000 --out data/fairness_balanced.csv

# Imbalanced demographics to test bias detection
python src/generate_data.py --n 40000 --seed 1001 --out data/fairness_imbalanced.csv
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   ```bash
   # For very large datasets, generate in chunks
   python src/generate_data.py --n 50000 --seed 1 --out data/chunk_1.csv
   python src/generate_data.py --n 50000 --seed 2 --out data/chunk_2.csv
   # Then combine: pandas.concat([pd.read_csv(f) for f in files])
   ```

2. **Reproducibility Issues**
   ```bash
   # Always use the same seed for reproducible results
   python src/generate_data.py --n 50000 --seed 42 --out data/reproducible.csv
   ```

3. **Data Quality Issues**
   ```python
   # Add validation after generation
   df = simulate(50000, 42)
   issues = validate_generated_data(df)
   if issues:
       print("Data quality issues found:", issues)
   ```

### Performance Optimization

```bash
# For faster generation on large datasets
python -O src/generate_data.py --n 1000000 --seed 42 --out data/large.csv
```

## Advanced Customizations

### Custom Risk Models

Create different risk calculation functions:

```python
def conservative_risk_model(df):
    """More conservative risk assessment"""
    # Implement stricter risk criteria
    pass

def aggressive_risk_model(df):
    """More lenient risk assessment"""
    # Implement looser risk criteria
    pass
```

### Industry-Specific Adaptations

```python
def food_delivery_focus(df):
    """Adapt data for food delivery specific metrics"""
    df["avg_delivery_time"] = RNG.normal(25, 8, len(df))
    df["food_safety_score"] = RNG.normal(4.2, 0.4, len(df))
    return df

def ride_hailing_focus(df):
    """Adapt data for ride-hailing specific metrics"""
    df["vehicle_cleanliness_score"] = RNG.normal(4.5, 0.3, len(df))
    df["route_efficiency_score"] = RNG.normal(0.85, 0.1, len(df))
    return df
```

## Data Export Formats

### Multiple Format Support

```python
# CSV (default)
df.to_csv("data/partners.csv", index=False)

# JSON
df.to_json("data/partners.json", orient="records")

# Parquet (efficient for large datasets)
df.to_parquet("data/partners.parquet")

# Excel (for business users)
df.to_excel("data/partners.xlsx", index=False)
```

---

## Quick Reference Commands

```bash
# Standard dataset
python src/generate_data.py --n 50000 --seed 42 --out data/partners.csv

# Small test dataset
python src/generate_data.py --n 1000 --seed 123 --out data/test.csv

# Large production dataset
python src/generate_data.py --n 200000 --seed 456 --out data/prod.csv

# Run full pipeline with custom data
python run_project.py
```

For more advanced customizations, modify the source files directly and refer to the inline comments in `src/generate_data.py` for guidance on parameter adjustments.
