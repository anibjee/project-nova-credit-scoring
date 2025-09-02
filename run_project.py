#!/usr/bin/env python3
"""
Project Nova - Complete Pipeline Runner
Run this script to execute the entire credit scoring pipeline
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd):
    """Run a shell command and handle errors"""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success!")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"❌ Error: {result.stderr}")
        return False
    return True

def main():
    print("=" * 70)
    print("🚀 PROJECT NOVA - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n📊 Step 1: Generating synthetic partner data...")
    if not run_command("python src/generate_data.py --n 50000 --seed 42 --out data/partners.csv"):
        sys.exit(1)
    
    # Step 2: Train baseline model
    print("\n🤖 Step 2: Training baseline model...")
    if not run_command("""python src/train_model.py ^
        --data data/partners.csv ^
        --model_out models/model_baseline.pkl ^
        --metrics_out reports/metrics_baseline.json ^
        --fairness_out reports/fairness_baseline.json ^
        --scores_out data/partners_scores_baseline.csv ^
        --mitigation none""".replace('\n        ', ' ')):
        sys.exit(1)
    
    # Step 3: Train fair model
    print("\n⚖️  Step 3: Training fairness-optimized model...")
    if not run_command("""python src/train_model.py ^
        --data data/partners.csv ^
        --model_out models/model_fair.pkl ^
        --metrics_out reports/metrics_fair.json ^
        --fairness_out reports/fairness_fair.json ^
        --scores_out data/partners_scores_fair.csv ^
        --mitigation equalized_odds""".replace('\n        ', ' ')):
        sys.exit(1)
    
    # Step 4: Completion message
    print("\n✅ Training complete!")
    
    print("\n" + "=" * 70)
    print("🎉 PROJECT NOVA PIPELINE COMPLETE!")
    print("📋 Results available:")
    print("   • Check 'reports/' for detailed metrics")
    print("   • View 'data/partners_scores_*.csv' for Nova scores")
    print("   • Models saved in 'models/' directory")
    print("=" * 70)

if __name__ == "__main__":
    main()
