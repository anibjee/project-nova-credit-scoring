# Installation Guide

> ✅ **TESTED SOLUTION**: Use `pip install -r requirements-minimal.txt` - this works reliably on Windows and other platforms!

## Quick Start

1. **Create virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate virtual environment:**
   
   **Windows (PowerShell):**
   ```bash
   .\.venv\Scripts\Activate.ps1
   ```
   
   **Windows (Command Prompt):**
   ```bash
   .venv\Scripts\activate.bat
   ```
   
   **Linux/Mac:**
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   
   **✅ RECOMMENDED (Works on Windows):**
   ```bash
   pip install -r requirements-minimal.txt
   ```
   

## If Installation Fails

### Option 1: Use minimal requirements
```bash
pip install -r requirements-minimal.txt
```

### Option 2: Install individually
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install jupyter fairlearn shap scipy joblib
```

### Option 3: Update pip first
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option 4: Use conda instead
```bash
conda install numpy pandas scikit-learn matplotlib seaborn jupyter scipy
pip install fairlearn shap
```

## Common Issues

### 1. Metadata generation error
**Solution:** Upgrade pip and setuptools
```bash
python -m pip install --upgrade pip setuptools wheel
```

### 2. C++ compiler error (Windows)
**Solution:** Install Visual Studio Build Tools or use pre-compiled packages
```bash
pip install --only-binary=all -r requirements.txt
```

### 3. Permission error
**Solution:** Use --user flag
```bash
pip install --user -r requirements.txt
```

## Verify Installation

Run this to check if everything is installed correctly:
```python
import numpy, pandas, sklearn, matplotlib, seaborn, fairlearn, shap
print("✅ All packages installed successfully!")
```

## Run the Project

Once installed, run:
```bash
python run_project.py
```
