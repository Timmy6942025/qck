#!/usr/bin/env python3
"""
BTC Money Maker - Live Trading Script
Run this to get signals from the model
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_model():
    with open("models/btc_money_maker.pkl", "rb") as f:
        return pickle.load(f)

def get_signal():
    """Get current trading signal"""
    # Load model
    artifacts = load_model()
    model = artifacts['model']
    features = artifacts['features']
    config = artifacts['config']
    
    # Load latest data (you'd connect to live data here)
    # For now just return config info
    return {
        'threshold': config['threshold'],
        'leverage': config['leverage'],
        'target': config['target_gain'],
        'expected_winrate': config['expected_winrate'],
        'message': 'Connect to live Binance API to get real signals'
    }

if __name__ == "__main__":
    signal = get_signal()
    print("="*50)
    print("BTC MONEY MAKER")
    print("="*50)
    for k, v in signal.items():
        print(f"{k}: {v}")
