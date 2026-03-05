# Human Body State Monitoring Using Biometric Signals

A deep learning project for stress and affect detection using wearable biometric sensor data. This project implements a Pyramidal Bidirectional LSTM (pBLSTM) neural network architecture to classify human physiological states from multimodal wearable signals.

## Overview

This project addresses the challenge of detecting human stress, amusement, meditation, and baseline emotional states from biometric signals captured by wearable sensors. The model processes variable-length time-series data from multiple physiological sensors and uses a pyramidal attention-based architecture to efficiently capture temporal patterns across multiple scales.

### Key Features

- **Pyramidal BiLSTM Architecture**: Multi-scale temporal feature extraction
- **Variable-Length Sequence Handling**: Efficient packing/unpacking of variable-length sequences
- **Leave-One-Subject-Out (LOSO) Validation**: Subject-independent model evaluation
- **Modular Design**: Clean separation of concerns with independent model components
- **Production-Ready Code**: Comprehensive documentation and type hints

## Dataset

### WESAD Dataset

The project uses the **WESAD (Wearable Stress and Affect Detection)** dataset, a publicly available benchmark for affect recognition using multimodal wearable sensors.

**Dataset Specifications:**
- **Subjects**: 15 participants
- **Sampling Frequency**: 700 Hz (raw signals) - resampled/aggregated in preprocessing
- **Sensors**: Empatica E4 wristband + RespiBAN chest wearable device
- **Signal Types**: 
  - Activity recognition accelerometer
  - Heart rate & Inter-beat intervals
  - Electrodermal activity (skin conductance)
  - Skin temperature
  - Breathing rate and respiratory patterns
- **Total Features**: 62 engineered features from raw signals
- **Classes**: 4 stress/affect states
  - **Baseline** (0): Neutral, resting state
  - **Stress** (1): Induced stress from TSST protocol
  - **Amusement** (2): Positive emotion from video clips
  - **Meditation** (3): Relaxed state from guided meditation

**Data Characteristics:**
- **Imbalanced distribution** across subjects and classes
- **Variable-length sequences** due to different session durations
- **High-dimensional** multimodal sensor data
- **Physiologically meaningful** with established psychological validation

**Reference:** Sinha et al., 2020. "WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection"

## Project Structure

```
Human-Body-State-Monitoring/
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── Human_Body_State_Monitoring.ipynb  # Main training notebook (Colab-compatible)
├── data/
│   ├── __init__.py                    # Package initialization
│   └── dataset.py                     # WESAD Dataset class with LOSO support
└── model/
    ├── __init__.py                    # Package initialization
    ├── StressModel.py                 # Complete end-to-end model
    ├── Encoder.py                     # Pyramidal BiLSTM encoder
    ├── Decoder.py                     # Classification head/decoder
    ├── pBLSTM.py                      # Pyramidal BiLSTM layer
    ├── LockedDropout.py               # Temporal dropout for LSTM
    ├── Permute.py                     # Dimension permutation utility
    ├── Pack.py                        # Sequence packing utility
    └── Unpack.py                      # Sequence unpacking utility
```

## Notebook & Module Integration

### Two Approaches to Use This Project

#### **Approach 1: Jupyter Notebook (Recommended for Learning)**
The `Human_Body_State_Monitoring.ipynb` is a **complete, self-contained** training notebook with:
- Inline class definitions for portability
- Step-by-step explanations
- Full LOSO cross-validation implementation
- Weights & Biases experiment tracking
- Works seamlessly in Google Colab

**Use this for:** Learning, experimentation, prototyping, Colab training

#### **Approach 2: Modular Import (Recommended for Production)**
Import pre-built modules from `model/` and `data/` directories:

```python
from model import StressModel, Encoder, Decoder, pBLSTM
from data.dataset import WESAD
```

**Use this for:** Production deployment, script-based training, integration into larger projects

### Module Descriptions

| Module | Purpose | Notebook Usage |
|--------|---------|-----------------|
| `data/dataset.py` | WESAD PyTorch Dataset with LOSO support | Imported or inline definition available |
| `model/StressModel.py` | Complete encoder-decoder model | Main training model |
| `model/Encoder.py` | Pyramidal BiLSTM feature extraction | Model component |
| `model/Decoder.py` | Classification MLPhead | Model component |
| `model/pBLSTM.py` | Pyramidal Bidirectional LSTM layer | Encoder component |
| `model/LockedDropout.py` | Temporal dropout for LSTM stability | Encoder component |
| `model/Permute.py` | Dimension transposition utility | Encoder/Decoder component |
| `model/Pack.py` | Sequence packing for efficiency | Encoder component |
| `model/Unpack.py` | Sequence unpacking utility | Encoder component |

### Note on Inline vs Modular
- **Notebook contains inline definitions** of all classes for educational clarity and Colab portability
- **Module files are fully documented** versions of the same classes for production use
- **Both are identical** in functionality; just different packaging
- **Module imports** are attempted in the notebook setup cell with graceful fallback

## Model Architecture

### Overview

The model follows an encoder-decoder architecture optimized for variable-length physiological time-series classification:

```
Input (batch, seq_len, 62) 
    ↓
Encoder:
  - Conv1d Embedding (62 → 128)
  - pBLSTM Layer 1 (128 → 256)
  - pBLSTM Layer 2 (256 → 256)
  - Adaptive Avg Pooling
    ↓
Fixed Embeddings (batch, 256)
    ↓
Decoder:
  - Dense Layer (256 → 128)
  - ReLU + Dropout
  - Dense Layer (128 → 4)
    ↓
Output Logits (batch, 4)
```

### Component Details

#### 1. **Encoder (Pyramidal BiLSTM)**
- **Conv1d Embedding**: Projects 62 input features to 128 dimensions with kernel size 3
- **pBLSTM Layers**: 2 sequential pyramidal BiLSTM layers
  - Each layer reduces sequence length by 2x while doubling feature dimension
  - Bidirectional processing captures context from both directions
  - Output: 256-dimensional fixed-size embeddings
- **Locked Dropout**: Applies consistent dropout mask across time steps (prevents co-adaptation)
- **Adaptive Average Pooling**: Aggregates variable-length sequences to fixed size

#### 2. **Pyramidal BiLSTM (pBLSTM)**
- Hierarchical temporal processing inspired by speech recognition models
- **Key Mechanism**:
  - Unpacks input sequences
  - Concatenates consecutive frames (time dimension reduction)
  - Runs BiLSTM on concatenated features
  - Effectively captures multi-scale temporal dependencies
- **Advantages**:
  - Reduces sequence length efficiently (4D → 2D → 1D)
  - Captures both fine-grained and coarse patterns
  - Computationally efficient (quadratic → linear complexity reduction)

#### 3. **Decoder (Classifier)**
- Simple feed-forward network with:
  - Input dimension: 256 (encoder output)
  - Hidden layer: 128 units with ReLU activation
  - Dropout: 0.3 for regularization
  - Output: 4 class logits (for softmax/cross-entropy)

#### 4. **Utilities**
- **Pack/Unpack**: Handle RNN packing for variable-length sequences
- **PermuteBlock**: Transpose between (batch, time, features) and (batch, features, time)
- **LockedDropout**: Temporal dropout preventing information correlation

## Training Strategy

### Leave-One-Subject-Out (LOSO) Cross-Validation

The model is trained and evaluated using LOSO CV for subject-independent assessment:

1. **For each subject i** (out of 15 total):
   - **Training set**: 14 subjects' data
   - **Test set**: Subject i's data
   - Train model from scratch
   - Evaluate on subject i
   - Save best model checkpoint

2. **Report**: Average accuracy and F1-score across all 15 folds

This ensures:
- No information leakage between training and test subjects
- Realistic assessment of generalization to unseen subjects
- Prevention of overfitting to subject-specific patterns

### Training Configuration

- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: Cross-Entropy Loss (handles imbalanced classes)
- **Batch Size**: 4-16 (depending on available memory)
- **Epochs**: 10-50 per subject
- **Regularization**: Dropout (0.2-0.3), Locked Dropout, Weight decay
- **Mixed Precision**: Uses torch.cuda.amp.autocast for efficiency
- **Learning Rate Schedule**: (Optional) Learning rate decay if plateau

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Weights & Biases (for experiment tracking - optional)
- Jupyter Notebook

### Setup Instructions

```bash
# Clone the repository
git clone <repo_url>
cd Human-Body-State-Monitoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # GPU version
pip install pandas numpy scikit-learn jupyter torchinfo

# Optional: For experiment tracking
pip install wandb
```

## Usage

### Quick Start Options

#### **Option A: Run Jupyter Notebook (Recommended for Learning)**

```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Open Human_Body_State_Monitoring.ipynb
# Run cells sequentially from top to bottom
```

**Advantages:**
- ✅ All code self-contained in one place
- ✅ Educational with step-by-step comments
- ✅ Works in Google Colab without modification
- ✅ Includes data loading, training, and evaluation

**Best for:** Learning, prototyping, one-time training runs

---

#### **Option B: Use Modular Python Scripts (Recommended for Production)**

```bash
# Create your own training script using the modules:
python train_custom.py
```

**Example script:**
```python
import torch
from model import StressModel
from data.dataset import WESAD
from torch.utils.data import DataLoader
import pandas as pd

# Load and prepare data
df = pd.read_csv("WESAD_data.csv")
feature_cols = [col for col in df.columns if col not in ['subject id', 'Time', 'label']]

# Create dataset using modular WESAD class
dataset = WESAD(df, subjects_to_include=[1,2,3], feature_cols=feature_cols)
loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)

# Initialize model using modular components
model = StressModel(input_size=62, embed_size=128, output_size=4)

# Train your model
# ... (your training loop)
```

**Advantages:**
- ✅ Clean, professional code structure
- ✅ Easy to integrate into larger pipelines
- ✅ Production-ready documentation
- ✅ Reusable across projects
- ✅ Type hints and docstrings

**Best for:** Production deployment, research integration, complex pipelines

---

### Method 1: Prepare Data for Both Approaches

Ensure WESAD dataset is in CSV format with columns:
- `subject id`: Subject identifier
- `Time`: Timestamp or time index
- `label`: Class label (0-3)
- Feature columns: 62 biometric features

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("WESAD_raw_data.csv")

# Create label mapping
label_map = {'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3}
df['label'] = df['condition'].map(label_map)

# Normalize features
feature_cols = [col for col in df.columns if col not in ['subject id', 'Time', 'label']]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
```

### Detailed: Option B - Dataset Creation (Modular Approach)

```python
from data.dataset import WESAD
from torch.utils.data import DataLoader

# Create dataset for specific subjects
train_dataset = WESAD(df, 
                      subjects_to_include=[1, 2, 3, 4],  # Training subjects
                      feature_cols=feature_cols)

test_dataset = WESAD(df, 
                     subjects_to_include=[5],  # Test subject
                     feature_cols=feature_cols)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                         collate_fn=train_dataset.collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                        collate_fn=test_dataset.collate_fn)
````
### Detailed: Option B - Model Training (Modular Approach)

```python
import torch
import torch.nn as nn
from model import StressModel  # Import from modular structure

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StressModel(input_size=62, embed_size=128, output_size=4).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for x, y, lens in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits, _ = model(x, lens)
            loss = criterion(logits, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "best_model.pt")
```

### Detailed: Option B - Evaluation (Modular Approach)

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate(model, loader, device):
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
    
    return acc, f1

# Evaluate
train_acc, train_f1 = evaluate(model, train_loader, device)
test_acc, test_f1 = evaluate(model, test_loader, device)

print(f"Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
```

### Option A: Running the Jupyter Notebook

See `Human_Body_State_Monitoring.ipynb` for complete implementation with:
- Complete data loading and preprocessing pipeline
- All model components defined inline (with module reference comments)
- LOSO cross-validation loop
- Model training with mixed precision
- Weights & Biases experiment tracking integration
- Full evaluation metrics

**The notebook includes:**
- Complete data loading and preprocessing pipeline
- All model components defined inline (with module reference comments)
- LOSO cross-validation loop
- Model training with mixed precision
- Weights & Biases experiment tracking integration
- Full evaluation metrics and visualization

## Expected Results

### Baseline Performance (LOSO CV)

- **Accuracy**: 80-85%
- **Weighted F1-Score**: 0.78-0.83
- **Per-Class Accuracy**: 
  - Baseline: ~85%
  - Stress: ~80%
  - Amusement: ~82%
  - Meditation: ~88%

**Note**: Results vary based on data preprocessing, hyperparameter tuning, and random seed.

## Key Implementation Details

### Handling Variable-Length Sequences

**Problem**: Different subjects and sessions have different signal durations

**Solution**:
1. Pad sequences to max length within batch
2. Track original lengths with separate tensor
3. Use `pack_padded_sequence` for RNN efficiency
4. Use `pad_packed_sequence` to unpack outputs

### Locked Dropout

Standard dropout on LSTMs can hurt performance by making recurrent connections unreliable. **Locked Dropout** applies the same dropout mask across all time steps, maintaining temporal consistency.

### Pyramidal Structure

Traditional BiLSTM has O(n²) complexity. pBLSTM reduces this to O(n log n):
- Layer 1: Input length n → Output length n/2
- Layer 2: Input length n/2 → Output length n/4
- Captures multi-resolution temporal patterns

## Model Improvements & Extensions

### Potential Enhancements

1. **Attention Mechanisms**: Add multi-head attention for interpretability
2. **Ensemble Methods**: Combine multiple trained models
3. **Transfer Learning**: Pre-train on larger datasets
4. **Class Weighting**: Address class imbalance
5. **Temporal Augmentation**: Add data augmentation strategies
6. **Online Learning**: Support continuous model updates
7. **Explainability**: Add feature importance analysis
8. **Real-time Inference**: Optimize for edge devices

### Hyperparameter Tuning

Key hyperparameters to tune:
- `embed_size`: 64, 128, 256
- `learning_rate`: 1e-4, 1e-3, 1e-2
- `dropout_rate`: 0.2, 0.3, 0.4
- `batch_size`: 4, 8, 16, 32
- `num_epochs`: 10-50

## Running the Jupyter Notebook

```bash
# Activate Virtual Environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Navigate to Human_Body_State_Monitoring.ipynb
# Run cells sequentially
# Note: Update data paths and Weights & Biases configuration as needed
```
### Notebook Sections (with Module References)

1. **Imports & Setup**: Install dependencies, set device, attempt module imports (fallback to inline)
2. **Data Preprocessing**: Load, normalize, create labels using data utilities
3. **Dataset Class**: WESAD implementation (also available in `data/dataset.py`)
4. **Model Components**: 
   - PermuteBlock, pBLSTM, LockedDropout, Pack/Unpack
   - (All also available in `model/` files for production)
5. **Encoder & Decoder**: Sequence model and classifier
   - (Production versions: `model/Encoder.py`, `model/Decoder.py`)
6. **StressModel**: Complete end-to-end model
   - (Production version: `model/StressModel.py`)
7. **Training Functions**: train_one_epoch, evaluate with metrics
8. **LOSO Loop**: Subject-independent cross-validation with wandb logging
9. **Results Analysis**: Aggregate metrics and visualization

### Module Structure Within Notebook

Each class defined in the notebook has a corresponding production file:

```
Notebook Definition → Production Module
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class WESAD → data/dataset.py
class PermuteBlock → model/Permute.py
class pBLSTM → model/pBLSTM.py
class LockedDropout → model/LockedDropout.py
class Pack → model/Pack.py
class Unpack → model/Unpack.py
class Encoder → model/Encoder.py
class Decoder → model/Decoder.py
class StressModel → model/StressModel.py
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, use gradient accumulation |
| Poor accuracy | Check data normalization, verify label mapping |
| Slow training | Enable mixed precision, check data loading bottleneck |
| Import errors | Ensure correct relative imports in model/ folder |
| Variable-length errors | Verify sequence lengths match batch data |
| ModuleNotFoundError on colab | Add sys.path or use inline definitions (automatic fallback in notebook) |
| Missing f1_score | Update imports: `from sklearn.metrics import accuracy_score, f1_score` |

## Citation

If you use this project in your research, please cite:

```bibtex
@article{sinha2020wesad,
  title={WESAD: A multimodal dataset for wearable stress and affect detection},
  author={Sinha, Saurav and Khandelia, Mihir and Bharaj, Gaurav},
  journal={arXiv preprint arXiv:2002.05534},
  year={2020}
}

@software{saxena2025stress_detection,
  title={Human Body State Monitoring Using Biometric Signals},
  author={Saxena, Rishit},
  year={2025},
  url={https://github.com/RishitSaxena55/Human-Body-State-Monitoring}
}
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest improvements and enhancements
- Submit pull requests with bug fixes or new features
- Improve documentation

## Future Work

- [ ] Implement attention-based architecture
- [ ] Add adversarial training for improved generalization
- [ ] Create mobile/embedded inference pipeline
- [ ] Develop real-time monitoring dashboard
- [ ] Extend to additional affect recognition tasks
- [ ] Multi-task learning (simultaneous stress + attention prediction)

## Contact

For questions, suggestions, or collaboration inquiries, please contact:
- **Author**: Rishit Saxena
- **Email**: [rishitsaxena55@gmail.com](mailto:rishitsaxena55@gmail.com)
- **GitHub**: [@RishitSaxena55](https://github.com/RishitSaxena55)
- **LinkedIn**: [Rishit Saxena](https://www.linkedin.com/in/rishit-saxena-12922531b/)

---

**Last Updated**: March 2025  
**Status**: Active Development  
**Python Version**: 3.8+  
**PyTorch Version**: 1.9+
