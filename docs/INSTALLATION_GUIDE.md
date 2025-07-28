# 🛠️ Complete Installation Guide

## 📋 **System Requirements**

### **Minimum Requirements**
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 or higher (3.9+ recommended)
- **Memory**: 8GB RAM minimum (16GB recommended for large datasets)
- **Storage**: 5GB free space (10GB for full development environment)
- **Network**: Stable internet connection for SAS environment access

### **Recommended Development Environment**
- **IDE**: VS Code, PyCharm, or Jupyter Lab
- **Python**: 3.9+ with virtual environment support
- **Memory**: 16GB+ RAM for optimal performance
- **Storage**: SSD for faster data processing
- **Network**: High-speed connection for SAS Cloud services

---

## 🔧 **Step-by-Step Installation**

### **Phase 1: Environment Setup**

#### **1.1 Python Environment**
```bash
# Check Python version
python --version  # Should be 3.8+

# Create isolated virtual environment
python -m venv sas_ml_env

# Activate environment
# Windows:
sas_ml_env\Scripts\activate
# macOS/Linux:
source sas_ml_env/bin/activate

# Verify activation
which python  # Should point to virtual environment
```

#### **1.2 Repository Setup**
```bash
# Clone repository
git clone <repository-url>
cd SAS_Model_Manager_Lifecycle

# Verify project structure
ls -la  # Should see main.py, src/, data/, etc.
```

### **Phase 2: Dependencies Installation**

#### **2.1 Core Dependencies**
```bash
# Upgrade pip for latest features
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(pandas|scikit-learn|swat|sasctl)"
```

#### **2.2 Development Dependencies (Optional)**
```bash
# For development and testing
pip install jupyter jupyterlab pytest black flake8

# For visualization enhancements
pip install plotly seaborn matplotlib

# For performance profiling
pip install memory-profiler line-profiler
```

#### **2.3 Dependency Verification**
```python
# Test imports (run in Python)
import pandas as pd
import sklearn
import swat
import sasctl
print("✅ All core dependencies installed successfully!")
```

### **Phase 3: SAS Environment Configuration**

#### **3.1 SAS Access Setup**
```bash
# Verify SAS environment accessibility
ping create.demo.sas.com

# Check SSL certificate requirements
curl -v https://create.demo.sas.com
```

#### **3.2 Certificate Installation**
```bash
# Download SAS certificate (if not included)
# 1. Visit: https://create.demo.sas.com
# 2. Download certificate bundle
# 3. Place in certificates/ directory

# Verify certificate
openssl x509 -in certificates/demo-rootCA-Intermidiates_4CLI.pem -text -noout
```

#### **3.3 OAuth2 Token Setup**
```bash
# Follow SAS OAuth2 authentication process
# See TOKEN_SETUP.md for detailed instructions

# Required steps:
# 1. Contact SAS Administrator for credentials
# 2. Visit SAS OAuth2 endpoint
# 3. Complete authentication flow
# 4. Generate and save access/refresh tokens locally
```

### **Phase 4: Data Setup**

#### **4.1 Dataset Verification**
```python
# Verify dataset exists and is readable
import pandas as pd
df = pd.read_csv('data/raw/bank_churn.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("✅ Dataset ready for processing")
```

#### **4.2 Directory Structure Creation**
```bash
# Create required directories
mkdir -p data/processed
mkdir -p models/trained
mkdir -p models/pzmm_packages
mkdir -p reports/model_performance
mkdir -p reports/business_insights
mkdir -p reports/sas_reports

# Verify structure
tree . -L 3  # Linux/macOS
# or
dir /s  # Windows
```

---

## 🧪 **Installation Verification**

### **Complete System Test**
```python
# Run this test script to verify everything works
# Save as test_installation.py

import sys
import os
import pandas as pd
from src.config import Config
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer

def test_installation():
    print("🧪 Running Installation Tests...")
    
    # Test 1: Python version
    assert sys.version_info >= (3, 8), "❌ Python 3.8+ required"
    print("✅ Python version check passed")
    
    # Test 2: Dependencies
    try:
        import pandas, numpy, sklearn, swat, sasctl
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    # Test 3: Data availability
    config = Config()
    data_path = os.path.join(config.DATA_RAW_PATH, 'bank_churn.csv')
    assert os.path.exists(data_path), "❌ Dataset not found"
    print("✅ Dataset available")
    
    # Test 4: Basic preprocessing
    df = pd.read_csv(data_path)
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_data(df)
    print("✅ Data preprocessing functional")
    
    # Test 5: Model training setup
    trainer = ModelTrainer()
    trainer.initialize_models()
    print("✅ Model training setup successful")
    
    print("\n🎉 Installation verification completed successfully!")
    return True

if __name__ == "__main__":
    test_installation()
```

```bash
# Run verification test
python test_installation.py
```

---

## 🔧 **Environment-Specific Setup**

### **Windows Setup**
```bash
# Install Visual C++ Build Tools (if needed)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Windows-specific dependencies
pip install pywin32

# Set environment variables
set PYTHONPATH=%PYTHONPATH%;%CD%\src
```

### **macOS Setup**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### **Linux Setup**
```bash
# Ubuntu/Debian dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL dependencies
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

## 🐳 **Docker Installation (Optional)**

### **Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src

# Run application
CMD ["python", "main.py"]
```

### **Docker Setup**
```bash
# Build image
docker build -t sas-ml-lifecycle .

# Run container
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models sas-ml-lifecycle

# For interactive development
docker run -it -v $(pwd):/app sas-ml-lifecycle bash
```

---

## 🚨 **Troubleshooting Installation Issues**

### **Common Problems & Solutions**

#### **Issue: SSL Certificate Errors**
```bash
# Problem: SSL verification failed
# Solution 1: Update certificates
pip install --upgrade certifi

# Solution 2: Download latest SAS certificate
curl -o certificates/sas-cert.pem https://create.demo.sas.com/path/to/cert

# Solution 3: Temporary workaround (not recommended for production)
export PYTHONHTTPSVERIFY=0
```

#### **Issue: Memory Errors During Installation**
```bash
# Problem: Out of memory during pip install
# Solution: Install with lower memory usage
pip install --no-cache-dir -r requirements.txt

# For large packages, install individually
pip install pandas numpy
pip install scikit-learn
pip install swat sasctl
```

#### **Issue: Permission Denied**
```bash
# Problem: Permission errors on Windows/macOS
# Solution: Install in user directory
pip install --user -r requirements.txt

# Or run with administrator privileges
sudo pip install -r requirements.txt  # Linux/macOS
# Run as Administrator on Windows
```

#### **Issue: SAS Connection Test Fails**
```python
# Problem: Cannot connect to SAS environment
# Solution: Step-by-step diagnosis

# Test 1: Network connectivity
import requests
response = requests.get("https://create.demo.sas.com")
print(f"Status: {response.status_code}")

# Test 2: SWAT connection
import swat
try:
    cas = swat.CAS("create.demo.sas.com", protocol="https")
    print("✅ SWAT connection successful")
except Exception as e:
    print(f"❌ SWAT connection failed: {e}")

# Test 3: sasctl connection
from sasctl import Session
try:
    session = Session("https://create.demo.sas.com")
    print("✅ sasctl connection successful")
except Exception as e:
    print(f"❌ sasctl connection failed: {e}")
```

---

## 📚 **Next Steps**

After successful installation:

1. **📖 Read the [User Guide](USER_GUIDE.md)** for detailed usage instructions
2. **🔒 Set up authentication** following [TOKEN_SETUP.md](../TOKEN_SETUP.md)
3. **🚀 Run your first model** with `python main.py`
4. **📊 Check sample outputs** in the `reports/` directory
5. **🔧 Customize configuration** in `src/config.py`

---

## 📞 **Support**

If you encounter installation issues:

1. **Check this guide** for common solutions
2. **Review system requirements** and verify compatibility
3. **Test individual components** using verification scripts
4. **Consult SAS documentation** for environment-specific issues
5. **Contact support** with detailed error logs

---

**🎉 You're ready to start building ML models with SAS integration!** 