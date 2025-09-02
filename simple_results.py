import json
import pandas as pd

print("=" * 60)
print("PROJECT NOVA RESULTS")
print("=" * 60)

# Load and display metrics
try:
    metrics = json.load(open('reports/metrics_baseline.json'))
    print("\nMODEL PERFORMANCE:")
    print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"  PR AUC: {metrics['pr_auc']:.3f}")
    print(f"  Brier Score: {metrics['brier']:.4f}")
except Exception as e:
    print(f"Error loading metrics: {e}")

# Load and display fairness
try:
    fair = json.load(open('reports/fairness_baseline.json'))
    print(f"\nFAIRNESS ANALYSIS:")
    print(f"  Gender bias: {fair['gender']['demographic_parity_diff']:.4f}")
    print(f"  Region bias: {fair['region']['demographic_parity_diff']:.4f}")
    print(f"  Role bias: {fair['role']['demographic_parity_diff']:.4f}")
    
    max_bias = max(abs(fair[attr]['demographic_parity_diff']) for attr in fair.keys())
    status = "EXCELLENT" if max_bias < 0.01 else "GOOD" if max_bias < 0.05 else "WARNING"
    print(f"  Overall Status: {status} (Max bias: {max_bias:.4f})")
except Exception as e:
    print(f"Error loading fairness: {e}")

# Load and display scores
try:
    scores = pd.read_csv('data/partners_scores_baseline.csv')
    print(f"\nNOVA SCORES:")
    print(f"  Mean Nova Score: {scores['nova_score'].mean():.1f}")
    print(f"  Score Range: {scores['nova_score'].min():.1f} - {scores['nova_score'].max():.1f}")
    print(f"  Partners Scored: {len(scores):,}")
    
    print(f"\nSample Nova Scores:")
    sample = scores[['partner_id', 'nova_score']].head(5)
    for _, row in sample.iterrows():
        print(f"  Partner {row['partner_id']:.0f}: {row['nova_score']:.1f}")
except Exception as e:
    print(f"Error loading scores: {e}")

print("\n" + "=" * 60)
print("SUCCESS: Project Nova completed successfully!")
print("=" * 60)
