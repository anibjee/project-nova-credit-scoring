#!/usr/bin/env python3
"""
Project Nova - Model Inference Demo
This script demonstrates how to use the trained model.pkl for predictions
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def load_model(model_path="models/model.pkl"):
    """Load the trained Project Nova model"""
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"   Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def create_sample_partner():
    """Create a sample partner for prediction demonstration"""
    return {
        'role': 'driver',
        'region': 'metro', 
        'gender': 'female',
        'age': 32,
        'tenure_months': 24,
        'earnings_monthly': 1200.0,
        'trips_weekly': 85.0,
        'on_time_rate': 0.92,
        'cancel_rate': 0.05,
        'customer_rating': 4.7,
        'safety_incidents_12m': 0,
        'wallet_txn_count_monthly': 95.0,
        'wallet_txn_value_monthly': 450.0,
        'income_volatility': 0.15,
        'seasonality_index': 1.1,
        'prior_loans': 0,
        'prior_defaults': 0,
        'vehicle_type': 'car',
        'fuel_cost_share': 0.18
    }

def nova_score_from_prob(prob_default):
    """Convert default probability to Nova Score (300-850)"""
    return 300.0 + (1.0 - prob_default) * 550.0

def predict_partner(model, partner_data):
    """Make a prediction for a single partner"""
    # Convert to DataFrame (model expects this format)
    partner_df = pd.DataFrame([partner_data])
    
    # Get probability of default
    prob_default = model.predict_proba(partner_df)[0, 1]
    
    # Get binary prediction
    prediction = model.predict(partner_df)[0]
    
    # Calculate Nova Score
    nova_score = nova_score_from_prob(prob_default)
    
    return {
        'probability_default': prob_default,
        'binary_prediction': prediction,
        'nova_score': nova_score
    }

def assess_risk(nova_score):
    """Provide risk assessment based on Nova Score"""
    if nova_score >= 750:
        return "🟢 Low Risk - Excellent creditworthiness"
    elif nova_score >= 650:
        return "🟡 Medium Risk - Good creditworthiness"
    elif nova_score >= 550:
        return "🟠 High Risk - Fair creditworthiness"
    else:
        return "🔴 Very High Risk - Poor creditworthiness"

def main():
    print("=" * 70)
    print("🔮 PROJECT NOVA - MODEL INFERENCE DEMO")
    print("=" * 70)
    
    # Check if model exists
    model_path = "models/model.pkl"
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Run 'python src/train_model.py --data data/partners.csv' first")
        return
    
    # Load model
    model = load_model(model_path)
    if model is None:
        return
    
    # Create sample partner
    print("\n📋 SAMPLE PARTNER PROFILE:")
    partner = create_sample_partner()
    for key, value in partner.items():
        print(f"   • {key}: {value}")
    
    # Make prediction
    print(f"\n🎯 MAKING PREDICTION...")
    try:
        result = predict_partner(model, partner)
        
        print(f"\n💎 NOVA SCORE RESULTS:")
        print(f"   • Default Probability: {result['probability_default']:.4f} ({result['probability_default']*100:.2f}%)")
        print(f"   • Binary Prediction: {'Default Risk' if result['binary_prediction'] else 'No Default'}")
        print(f"   • Nova Score: {result['nova_score']:.0f}")
        print(f"   • Risk Assessment: {assess_risk(result['nova_score'])}")
        
        # Show what this means for lending
        print(f"\n💰 LENDING IMPLICATIONS:")
        if result['nova_score'] >= 750:
            print("   ✅ Pre-approved for premium financial products")
            print("   ✅ Lowest interest rates available")
            print("   ✅ Higher credit limits")
        elif result['nova_score'] >= 650:
            print("   ✅ Approved for standard financial products")
            print("   ✅ Competitive interest rates")
            print("   ✅ Moderate credit limits")
        elif result['nova_score'] >= 550:
            print("   ⚠️  Conditional approval with higher rates")
            print("   ⚠️  May require additional documentation")
            print("   ⚠️  Lower initial credit limits")
        else:
            print("   ❌ Higher risk - specialized products only")
            print("   ❌ Requires manual review")
            print("   ❌ Secured credit products recommended")
            
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
    
    print(f"\n" + "=" * 70)
    print("🎉 Demo complete!")
    print("💡 This model can now be used for real-time credit scoring")
    print("=" * 70)

if __name__ == "__main__":
    main()
