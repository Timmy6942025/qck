"""
Deep Learning Model for BTC Short-Term Trading
Uses LSTM/GRU with comprehensive feature engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from pathlib import Path
import pickle
from datetime import datetime, timedelta

# Configuration
@dataclass
class ModelConfig:
    # Model architecture
    sequence_length: int = 60  # 60 minutes of history
    horizon: int = 5  # Predict 5 minutes ahead
    
    # LSTM config
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 20
    early_stopping_patience: int = 5
    
    # Data
    train_bars: int = 100000  # Use last N bars for training
    test_bars: int = 10000
    
    # Trading costs
    fee_bps: float = 4
    slippage_bps: float = 2


def load_data() -> pd.DataFrame:
    """Load OHLCV and funding rate data"""
    data_dir = Path("/Users/Timothy/qck/data/processed")
    
    # Load main OHLCV data
    df = pd.read_parquet(data_dir / "spot_BTCUSDT_1m_full.parquet")
    print(f"Loaded OHLCV: {len(df):,} rows")
    
    # Load funding rates
    try:
        funding = pd.read_parquet(data_dir / "funding_rates.parquet")
        print(f"Loaded funding rates: {len(funding):,} records")
    except Exception as e:
        print(f"Could not load funding rates: {e}")
        funding = None
    
    return df, funding


def build_features(df: pd.DataFrame, funding: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build comprehensive features for the model"""
    data = df.copy()
    
    # Ensure sorted
    data = data.sort_values("open_time").reset_index(drop=True)
    
    # ============= PRICE RETURNS =============
    for lag in [1, 3, 5, 10, 15, 30, 60]:
        data[f'ret_{lag}'] = data['close'].pct_change(lag)
    
    # ============= VOLATILITY =============
    for window in [5, 15, 30, 60]:
        data[f'volatility_{window}'] = data['ret_1'].rolling(window).std()
    
    # ============= VOLUME FEATURES =============
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    data['volume_z'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
    
    # ============= PRICE FEATURES =============
    # VWAP
    data['vwap'] = data['quote_volume'] / data['volume'].replace(0, np.nan)
    data['close_vs_vwap'] = (data['close'] / data['vwap']) - 1
    
    # High-Low range
    data['range'] = (data['high'] - data['low']) / data['close']
    data['range_ma'] = data['range'].rolling(20).mean()
    
    # Close position in range
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # ============= TECHNICAL INDICATORS =============
    # RSI
    delta = data['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # Moving averages
    for window in [9, 21, 50, 200]:
        data[f'sma_{window}'] = data['close'].rolling(window).mean()
        data[f'close_vs_sma_{window}'] = (data['close'] / data[f'sma_{window}']) - 1
    
    # Trend strength
    data['trend_10_30'] = (data['close'].rolling(10).mean() / data['close'].rolling(30).mean()) - 1
    
    # Bollinger Bands
    bb_mean = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_upper'] = bb_mean + 2 * bb_std
    data['bb_lower'] = bb_mean - 2 * bb_std
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower']).replace(0, np.nan)
    
    # ============= TIME FEATURES =============
    data['hour_of_day'] = data['open_time'].dt.hour
    data['day_of_week'] = data['open_time'].dt.dayofweek
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # ============= FUNDING RATE FEATURES =============
    if funding is not None:
        # Forward-fill funding rates to minute level
        funding_df = funding[['fundingTime', 'fundingRate']].copy()
        funding_df = funding_df.rename(columns={'fundingTime': 'open_time'})
        data = data.merge(funding_df, on='open_time', how='left')
        data['fundingRate'] = data['fundingRate'].ffill()
        data['fundingRate'] = data['fundingRate'].bfill()
        
        # Funding rate features
        data['funding_ma_7'] = data['fundingRate'].rolling(7).mean()  # 7-period MA
        data['funding_deviation'] = data['fundingRate'] - data['funding_ma_7']
    else:
        data['fundingRate'] = 0
        data['funding_ma_7'] = 0
        data['funding_deviation'] = 0
    
    # ============= TARGET =============
    # Predict if price goes up by more than trading costs
    config = ModelConfig()
    threshold = (config.fee_bps + config.slippage_bps) * 2 / 10000
    data['future_return'] = data['close'].shift(-config.horizon) / data['close'] - 1
    data['target'] = (data['future_return'] > threshold).astype(int)
    
    return data


# Define feature columns
FEATURE_COLUMNS = [
    # Returns
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'ret_15', 'ret_30', 'ret_60',
    # Volatility
    'volatility_5', 'volatility_15', 'volatility_30', 'volatility_60',
    # Volume
    'volume_ratio', 'volume_z',
    # Price features
    'close_vs_vwap', 'range', 'range_ma', 'close_position',
    # Technical
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'close_vs_sma_9', 'close_vs_sma_21', 'close_vs_sma_50', 'close_vs_sma_200',
    'trend_10_30', 'bb_position',
    # Time
    'hour_of_day', 'day_of_week', 'is_weekend',
    # Funding
    'fundingRate', 'funding_ma_7', 'funding_deviation'
]


class SequenceDataset(Dataset):
    """PyTorch dataset for sequences"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM output: (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden)
        
        # Classification
        output = self.classifier(context)
        return output.squeeze()


def create_sequences(data: np.ndarray, targets: np.ndarray, seq_length: int) -> tuple:
    """Create sequences for LSTM"""
    sequences = []
    seq_targets = []
    
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        seq_targets.append(targets[i])
    
    return np.array(sequences), np.array(seq_targets)


def train_model():
    """Main training function"""
    print("="*60)
    print("BTC LSTM Trading Model Training")
    print("="*60)
    
    # Check for MPS (Apple Silicon GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df, funding = load_data()
    
    # Build features
    print("\nBuilding features...")
    data = build_features(df, funding)
    
    # Drop NaN rows
    data = data.dropna(subset=FEATURE_COLUMNS + ['target']).reset_index(drop=True)
    print(f"After feature engineering: {len(data):,} rows")
    
    # Extract features and target
    X = data[FEATURE_COLUMNS].values
    y = data['target'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (use most recent for test)
    config = ModelConfig()
    train_size = len(X_scaled) - config.test_bars
    
    X_train = X_scaled[:train_size]
    y_train = y[:train_size]
    X_test = X_scaled[train_size:]
    y_test = y[train_size:]
    
    print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"Train target distribution: {np.mean(y_train):.3f}")
    print(f"Test target distribution: {np.mean(y_test):.3f}")
    
    # Create sequences
    print("\nCreating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, config.sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, config.sequence_length)
    
    print(f"Sequence shape: {X_train_seq.shape}")
    
    # Create datasets
    train_dataset = SequenceDataset(X_train_seq, y_train_seq)
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = LSTMModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    best_test_acc = 0
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluation
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                test_preds.extend(outputs.cpu().numpy())
                test_targets.extend(y_batch.numpy())
        
        test_preds = np.array(test_preds)
        test_targets = np.array(test_targets)
        
        # Calculate accuracy at threshold 0.5
        train_acc = ((outputs > 0.5).float() == y_batch).float().mean().item()
        test_acc = np.mean((test_preds > 0.5) == test_targets)
        
        # Calculate AUC-ROC
        from sklearn.metrics import roc_auc_score
        try:
            test_auc = roc_auc_score(test_targets, test_preds)
        except:
            test_auc = 0.5
        
        print(f"Epoch {epoch+1:2d}/{config.epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Test Acc: {test_acc:.3f} | Test AUC: {test_auc:.3f}")
        
        # Learning rate scheduling
        scheduler.step(1 - test_acc)
        
        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    final_preds = []
    final_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            final_preds.extend(outputs.cpu().numpy())
            final_targets.extend(y_batch.numpy())
    
    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)
    
    # Calculate final metrics
    final_acc = np.mean((final_preds > 0.5) == final_targets)
    from sklearn.metrics import roc_auc_score, classification_report
    final_auc = roc_auc_score(final_targets, final_preds)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Test Accuracy: {final_acc:.3f}")
    print(f"Test AUC-ROC: {final_auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(final_targets, final_preds > 0.5, target_names=['Down/Flat', 'Up']))
    
    # Save model and artifacts
    output_dir = Path("/Users/Timothy/qck/models")
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), output_dir / "lstm_model.pt")
    
    # Save scaler and config
    artifacts = {
        'scaler': scaler,
        'config': config,
        'feature_columns': FEATURE_COLUMNS,
        'test_accuracy': final_acc,
        'test_auc': final_auc
    }
    
    with open(output_dir / "lstm_artifacts.pkl", 'wb') as f:
        pickle.dump(artifacts, f)
    
    print(f"\nModel saved to {output_dir}")
    print(f"Features used: {len(FEATURE_COLUMNS)}")
    
    return model, artifacts


if __name__ == "__main__":
    train_model()
