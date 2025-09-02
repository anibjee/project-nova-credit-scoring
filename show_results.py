#!/usr/bin/env python3
import json
import pandas as pd

print("=" * 60)
print("🚀 PROJECT NOVA RESULTS")
print("=" * 60)

# Load fairness results
try:
    fair = json.load(open('reports/fairness_baseline.json'))
    print("\n📊 FAIRNESS ANALYSIS (Baseline Model):")
    print(f"  • Gender bias: {fair['gender']['demographic_parity_diff']:.4f}")
    print(f"  • Region bias: {fair['region']['demographic_parity_diff']:.4f}")
    print(f"  • Role bias: {fair['role']['demographic_parity_diff']:.4f}")
    
    max_bias = max(abs(fair[attr]['demographic_parity_diff']) for attr in fair.keys())
    if max_bias < 0.01:
        status = "✅ EXCELLENT"
    elif max_bias < 0.05:
        status = "✅ GOOD"
    else:
        status = "⚠️  WARNING"
    print(f"  • Overall Status: {status} (Max bias: {max_bias:.4f})")
except:
    print("\n❌ Fairness baseline report not found")

# Load model metrics
try:
    metrics = json.load(open('reports/metrics_baseline.json'))
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"  • ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"  • PR AUC: {metrics['pr_auc']:.3f}")
    print(f"  • Brier Score: {metrics['brier']:.4f}")
except:
    print("\n❌ Metrics baseline report not found")

# Load Nova scores
try:
    scores = pd.read_csv('data/partners_scores_baseline.csv')
    print(f"\n💎 NOVA SCORES:")
    print(f"  • Mean Nova Score: {scores['nova_score'].mean():.1f}")
    print(f"  • Score Range: {scores['nova_score'].min():.1f} - {scores['nova_score'].max():.1f}")
    print(f"  • Partners Scored: {len(scores):,}")
    
    print(f"\n📋 Sample Nova Scores:")
    sample = scores[['partner_id', 'nova_score']].head(5)
    for _, row in sample.iterrows():
        print(f"  • Partner {row['partner_id']}: {row['nova_score']:.1f}")
except:
    print("\n❌ Nova scores not found")

print("\n" + "=" * 60)
print("🎉 Project Nova is running successfully!")
print("💡 Next: Open Jupyter notebook for detailed analysis")
print("=" * 60)
