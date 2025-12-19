# Credit Card Fraud Detection

End-to-end pipeline for detecting fraudulent credit card transactions using classical preprocessing (scaling + SMOTE) and an MLP model. The project includes data download, preprocessing, notebooks for exploration and training, and saved models and results.

## Project Structure

```
requirements.txt
data/
  creditcard.csv
  train_data.pkl
  test_data.pkl
  scaler.pkl
models/
  best_model.h5
  fraud_detector_model.h5
notebooks/
  data_exploration.ipynb
  Credit_Card_Fraud_MLP.ipynb
results/
  eda_summary.txt
  evaluation_metrics.txt
scripts/
  download_data.py
  preprocess.py
```

## Setup (Windows)

1) Create and activate a virtual environment

```powershell
# From repo root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) Optional: Install Jupyter kernel for this venv

```powershell
python -m ipykernel install --user --name card-fraud-venv --display-name "CardFraud venv"
```

## Data

You need the Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`). Place it under `data/creditcard.csv`.

- Option A — Use the Kaggle API via helper script:

```powershell
python scripts\download_data.py
```

The script guides you to set `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables (or a `.env` file) and downloads/unzips the dataset to `data/`.

- Option B — Manual download:
  1. Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  2. Extract and copy `creditcard.csv` to `data/creditcard.csv`

## Preprocessing

Run the preprocessing pipeline to split data, scale features, and apply SMOTE to the training set only. Outputs pickled train/test sets and scaler in `data/`.

```powershell
python scripts\preprocess.py
```

Artifacts produced:

- `data/train_data.pkl` (SMOTE-balanced training features/labels)
- `data/test_data.pkl` (scaled test features/labels, original distribution)
- `data/scaler.pkl` (fitted `StandardScaler`)

## Notebooks

- Exploration: `notebooks/data_exploration.ipynb`
- Training: `notebooks/Credit_Card_Fraud_MLP.ipynb`

### Run locally

```powershell
jupyter notebook
# or
code .  # open VS Code and run notebooks via the Jupyter extension
```

Ensure your kernel is the venv created above (e.g., "CardFraud venv"). The training notebook can load the preprocessed artifacts from `data/` and save models to `models/`.

### Run in Google Colab

1. Upload `notebooks/Credit_Card_Fraud_MLP.ipynb` to Colab
2. Runtime → Change runtime type → Select GPU (optional)
3. Make data available:
   - Upload `data/train_data.pkl`, `data/test_data.pkl`, `data/scaler.pkl` via Colab's file sidebar, or
   - Mount Google Drive and copy the files there
4. Run all cells; models and metrics can be downloaded or saved to Drive

## Local vs Colab Training (GPU/CPU)

TensorFlow on Windows supports NVIDIA CUDA GPUs. Integrated Intel Iris Xe graphics are not supported by standard TensorFlow GPU builds; training will typically run on CPU.

Use the snippet below to check GPU availability and fall back to CPU. Note: `tf.test.is_gpu_available()` is deprecated in recent TF versions, but shown here for Iris Xe checks; a modern alternative is `tf.config.list_physical_devices('GPU')`.

```python
import os
import tensorflow as tf

print("TensorFlow:", tf.__version__)

# Deprecated API (often returns False on Iris Xe)
try:
    gpu_ok = tf.test.is_gpu_available()
    print("tf.test.is_gpu_available():", gpu_ok)
except AttributeError:
    gpu_ok = False

# Modern check
gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)

if not gpus:
    # Force CPU to avoid accidental GPU code paths
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("No compatible GPU found — using CPU.")
else:
    print("GPU available — using GPU.")
```

Tips:

- For Iris Xe, expect CPU training; consider smaller batch sizes and enabling mixed precision only on supported GPUs.
- On Colab (with GPU runtime), the check will show one or more GPUs (typically NVIDIA). Training will be faster.

## Models and Results

- Trained models: `models/best_model.h5`, `models/fraud_detector_model.h5`
- Metrics: `results/evaluation_metrics.txt`
- EDA summary: `results/eda_summary.txt`

### Quick inference example (local)

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load artifacts
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)['scaler']

model = load_model('models/best_model.h5')

# Example: predict on first 10 test samples
with open('data/test_data.pkl', 'rb') as f:
    test = pickle.load(f)

X_test = test['X_test']
y_test = test['y_test']

pred_probs = model.predict(X_test.iloc[:10])
pred_labels = (pred_probs.flatten() > 0.5).astype(int)
print("Pred:", pred_labels)
print("True:", np.array(y_test.iloc[:10]))
```

## Quickstart

```powershell
# 1) Create venv and install deps
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Get data (Kaggle API)
python scripts\download_data.py

# 3) Preprocess
python scripts\preprocess.py

# 4) Explore & Train
jupyter notebook  # open notebooks/data_exploration.ipynb and Credit_Card_Fraud_MLP.ipynb
```

## Troubleshooting

- Kaggle API: Ensure `KAGGLE_USERNAME` and `KAGGLE_KEY` are set. In PowerShell:

```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_api_token"
```

- Memory constraints: Use smaller batches, consider downsampling for quick iterations.
- GPU issues on Windows: Without an NVIDIA GPU/CUDA, TensorFlow runs on CPU; this is expected for Intel Iris Xe.
