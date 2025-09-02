# 💎 Nova Score: Complete Location Guide

## 🎯 **What is the Nova Score?**

The **Nova Score** is Project Nova's proprietary credit score (300-850 scale) that converts default probability into a traditional credit score format:

**Formula**: `Nova Score = 300 + (1 - probability_of_default) × 550`

- **Higher Score = Better Creditworthiness**
- **Lower Score = Higher Risk**

---

## 📍 **WHERE TO FIND NOVA SCORES**

### 1. **📂 Data Files (Generated Scores)**

#### **Primary Nova Score File:**
```
data/partners_scores_baseline.csv
```
**Columns:**
- `partner_id`: Unique partner identifier
- `prob_default`: Probability of default (0-1)
- `nova_score`: Nova Credit Score (300-850)

#### **Sample Data:**
```csv
partner_id,prob_default,nova_score
21282,0.04527,825.1
12409,0.07585,808.3
31835,0.06779,812.7
8538,0.03083,833.0
```

#### **Additional Score Files:**
- `data/partners_scores_fair.csv` (Fair model scores)
- `data/partners_with_scores.csv` (Original scores)

### 2. **🔧 Source Code Implementation**

#### **Core Formula (src/train_model.py):**
```python
# Line 95-97: Nova Score Calculation Function
def to_nova_score(prob_default: np.ndarray) -> np.ndarray:
    # Map probability of default (higher worse) to credit score 300-850 (higher better)
    return 300.0 + (1.0 - prob_default) * 550.0

# Line 168: Applied during training
nova = to_nova_score(y_prob)

# Line 186: Saved to CSV
out_df["nova_score"] = nova
```

#### **Inference Implementation (predict_demo.py):**
```python
# Line 47-49: Prediction function
def nova_score_from_prob(prob_default):
    """Convert default probability to Nova Score (300-850)"""
    return 300.0 + (1.0 - prob_default) * 550.0

# Line 63: Used in predictions
nova_score = nova_score_from_prob(prob_default)
```

### 3. **📊 Live Outputs**

#### **When you run: `python simple_results.py`**
```
NOVA SCORES:
  Mean Nova Score: 807.4
  Score Range: 525.6 - 850.0
  Partners Scored: 12,500

Sample Nova Scores:
  Partner 21282: 825.1
  Partner 12409: 808.3
  Partner 31835: 812.7
```

#### **When you run: `python predict_demo.py`**
```
💎 NOVA SCORE RESULTS:
   • Default Probability: 0.0462 (4.62%)
   • Binary Prediction: No Default
   • Nova Score: 825
   • Risk Assessment: 🟢 Low Risk - Excellent creditworthiness
```

---

## 📈 **NOVA SCORE STATISTICS**

**From 12,500 partners scored:**

| Metric | Value |
|--------|-------|
| **Mean** | 807.4 |
| **Minimum** | 525.6 |
| **Maximum** | 850.0 |
| **Std Dev** | 34.7 |
| **Median** | 818.8 |

**Score Ranges:**
- **750-850**: Excellent creditworthiness (premium products)
- **650-749**: Good creditworthiness (standard products)
- **550-649**: Fair creditworthiness (conditional approval)
- **300-549**: Poor creditworthiness (high-risk products)

---

## 🔮 **HOW TO ACCESS NOVA SCORES**

### **Method 1: Load from CSV**
```python
import pandas as pd
scores = pd.read_csv('data/partners_scores_baseline.csv')
print(scores[['partner_id', 'nova_score']].head())
```

### **Method 2: Generate New Scores**
```python
import joblib
model = joblib.load('models/model_baseline.pkl')

# For new partner data
partner_data = {...}  # Partner features
prob_default = model.predict_proba([partner_data])[0, 1]
nova_score = 300 + (1 - prob_default) * 550
```

### **Method 3: Using Demo Script**
```bash
python predict_demo.py
```

---

## 🎯 **BUSINESS USE CASES**

### **Credit Approval:**
- **Score ≥ 750**: Auto-approve, lowest rates
- **Score 650-749**: Standard approval, competitive rates
- **Score 550-649**: Manual review, higher rates
- **Score < 550**: Declined or secured products

### **Risk Pricing:**
```python
def get_interest_rate(nova_score):
    if nova_score >= 750:
        return 0.05  # 5% APR
    elif nova_score >= 650:
        return 0.08  # 8% APR
    elif nova_score >= 550:
        return 0.12  # 12% APR
    else:
        return 0.18  # 18% APR
```

---

## 🎉 **SUMMARY**

**Nova Scores are found in:**
✅ **CSV files**: `data/partners_scores_*.csv` (12,500 scores)
✅ **Source code**: `src/train_model.py` + `predict_demo.py`
✅ **Live outputs**: Results from running scripts
✅ **Real-time**: Load model and predict for new partners

The Nova Score is the **core output** of the Project Nova credit scoring system!
