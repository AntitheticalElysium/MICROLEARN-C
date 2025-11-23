# MICROLEARN-C

Transpile scikit-learn models to standalone C code for microcontrollers.

## Features

- Linear Regression
- Logistic Regression  
- Decision Trees
- Zero dependencies
- Small binary size (~16KB)

## Usage

Train models:
```bash
python train_models.py
```

Transpile to C:
```bash
python transpile_simple_model.py models/linear_model.joblib
```

Run:
```bash
./model_inference
```

## Structure

- `c_models/` - C implementations
- `train_models.py` - Train all model types
- `transpile_simple_model.py` - Transpiler tool
- `models/` - Trained models

## Requirements

- Python 3.7+
- scikit-learn
- numpy
- joblib
- gcc
