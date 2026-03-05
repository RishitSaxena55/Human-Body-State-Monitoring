"""
Example training script using the modular components.
This demonstrates production-ready usage of the WESAD model.

Usage:
    python train_example.py --batch_size 8 --epochs 20 --lr 1e-3
"""

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

# Import from modular structure
from model import StressModel
from data.dataset import WESAD


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train WESAD stress detection model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="WESAD_raw_data.csv",
        help="Path to WESAD CSV file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--test_subject",
        type=int,
        default=1,
        help="Subject ID to use for testing (LOSO)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_data(data_path):
    """Load and preprocess WESAD dataset.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        df: Preprocessed DataFrame
        feature_cols: List of feature column names
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Create label mapping
    label_map = {'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3}
    df['label'] = df['condition'].map(label_map)
    
    # Drop unnecessary columns
    df = df.drop(columns=['condition', 'SSSQ'])
    
    # Extract feature columns
    feature_cols = [col for col in df.columns if col not in ['subject id', 'Time', 'label']]
    print(f"Found {len(feature_cols)} features")
    
    # Normalize features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    print(f"Data shape: {df.shape}")
    print(f"Subjects: {df['subject id'].nunique()}")
    print(f"Classes: {sorted(df['label'].unique())}")
    
    return df, feature_cols


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (x, y, lens) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(x, lens)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def evaluate(model, loader, device):
    """Evaluate model performance.
    
    Args:
        model: Neural network model
        loader: Data loader (train or test)
        device: Device to evaluate on
        
    Returns:
        Tuple of (accuracy, f1_score, confusion_matrix)
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for x, y, lens in loader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x, lens)
            pred = torch.argmax(logits, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return acc, f1, cm


def main():
    """Main training script."""
    args = parse_args()
    set_seed(args.seed)
    
    print(f"Using device: {args.device}")
    
    # Load data
    df, feature_cols = load_and_preprocess_data(args.data_path)
    
    # Prepare datasets (LOSO: Leave-One-Subject-Out)
    all_subjects = sorted(df['subject id'].unique())
    test_subject = args.test_subject
    train_subjects = [s for s in all_subjects if s != test_subject]
    
    print(f"\nLOSO Setup:")
    print(f"  Test subject: {test_subject}")
    print(f"  Training subjects: {train_subjects}")
    
    # Create datasets
    train_dataset = WESAD(df, subjects_to_include=train_subjects, feature_cols=feature_cols)
    test_dataset = WESAD(df, subjects_to_include=[test_subject], feature_cols=feature_cols)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # Initialize model
    input_size = len(feature_cols)
    model = StressModel(input_size=input_size, embed_size=128, output_size=4)
    model = model.to(args.device)
    
    print(f"\nModel initialized:")
    print(f"  Input size: {input_size}")
    print(f"  Embed size: 128")
    print(f"  Output classes: 4")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_f1 = 0.0
    best_model_state = None
    
    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}\n")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        
        # Evaluate
        train_acc, train_f1, _ = evaluate(model, train_loader, args.device)
        test_acc, test_f1, test_cm = evaluate(model, test_loader, args.device)
        
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best F1: {best_f1:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print(f"\nFinal Evaluation (Best Model):")
    test_acc, test_f1, test_cm = evaluate(model, test_loader, args.device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_cm)
    
    # Save model
    model_path = f"model_subject_{test_subject}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    return test_acc, test_f1


if __name__ == "__main__":
    main()
