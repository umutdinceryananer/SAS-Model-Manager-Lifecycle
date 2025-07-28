# 🔧 API Reference Guide

## 📚 **Module Overview**

This reference provides detailed documentation for all classes and methods in the SAS Model Lifecycle platform.

---

## 📊 **config.py**

### **Class: Config**
Central configuration management for the entire platform.

```python
class Config:
    """Configuration settings for the ML pipeline and SAS integration."""
```

#### **Attributes**
```python
# Path Configuration
PROJECT_ROOT: str           # Root directory of the project
DATA_RAW_PATH: str         # Raw data directory path
DATA_PROCESSED_PATH: str   # Processed data directory path
MODELS_PATH: str           # Model artifacts directory path
REPORTS_PATH: str          # Reports output directory path

# Model Parameters
TEST_SIZE: float = 0.2              # Train/test split ratio
RANDOM_STATE: int = 42              # Random seed for reproducibility
TARGET_COLUMN: str = 'Attrition_Flag'  # Target variable name

# Data Cleaning
COLUMNS_TO_DROP: List[str]          # Columns to remove during preprocessing

# SAS Configuration
SAS_CONFIG: Dict[str, Any]          # SAS connection settings
MODEL_MANAGER_CONFIG: Dict[str, Any] # Model Manager specific settings
```

#### **Usage Example**
```python
from src.config import Config

config = Config()
print(f"Data path: {config.DATA_RAW_PATH}")
print(f"Test size: {config.TEST_SIZE}")
print(f"SAS host: {config.SAS_CONFIG['hostname']}")
```

---

## 🔍 **exploratory_analysis.py**

### **Class: ExploratoryAnalyzer**
Handles exploratory data analysis and data profiling.

```python
class ExploratoryAnalyzer:
    """Performs comprehensive exploratory data analysis."""
```

#### **Methods**

##### **load_data(file_path: str) → pd.DataFrame**
Load and validate dataset from CSV file.

```python
def load_data(self, file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with validation.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data validation fails
    """
```

**Example:**
```python
analyzer = ExploratoryAnalyzer()
df = analyzer.load_data('data/raw/bank_churn.csv')
print(f"Loaded {len(df)} records")
```

##### **basic_info(df: pd.DataFrame) → pd.DataFrame**
Generate basic dataset information and statistics.

```python
def basic_info(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Display basic dataset information including:
    - Shape and memory usage
    - Data types distribution
    - Missing values analysis
    - Basic statistical summary
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Enhanced dataframe with analysis printed
    """
```

##### **target_analysis(df: pd.DataFrame) → pd.DataFrame**
Analyze target variable distribution and relationships.

```python
def target_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform target variable analysis:
    - Class distribution
    - Imbalance ratio
    - Target vs features correlation
    
    Args:
        df (pd.DataFrame): Input dataframe with target column
        
    Returns:
        pd.DataFrame: Dataframe with target analysis completed
    """
```

---

## 🔧 **data_preprocessing.py**

### **Class: DataPreprocessor**
Comprehensive data preprocessing and feature engineering pipeline.

```python
class DataPreprocessor:
    """5-stage data preprocessing pipeline for ML model preparation."""
```

#### **Attributes**
```python
config: Config                      # Configuration object
label_encoders: Dict[str, LabelEncoder]  # Fitted label encoders
scaler: StandardScaler              # Fitted feature scaler
feature_names: List[str]            # Final feature names after preprocessing
```

#### **Methods**

##### **clean_data(df: pd.DataFrame) → pd.DataFrame**
Stage 1: Data cleaning and preparation.

```python
def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare raw data:
    - Remove unnecessary columns
    - Convert target variable to binary
    - Handle multicollinearity
    - Validate data types
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
```

##### **feature_engineering(df: pd.DataFrame) → pd.DataFrame**
Stage 2: Advanced feature engineering.

```python
def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features:
    - Age grouping
    - Utilization categories
    - Interaction features
    - Risk scores
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe
    """
```

##### **encode_categorical_features(df: pd.DataFrame) → pd.DataFrame**
Stage 3: Categorical variable encoding.

```python
def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables:
    - Label encoding for ordinal features
    - One-hot encoding for nominal features
    - Target encoding for high-cardinality features
    
    Args:
        df (pd.DataFrame): Dataframe with categorical features
        
    Returns:
        pd.DataFrame: Encoded dataframe
    """
```

##### **scale_numerical_features(df: pd.DataFrame, fit_scaler: bool = False) → pd.DataFrame**
Stage 4: Numerical feature scaling.

```python
def scale_numerical_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler:
    - Z-score normalization
    - Mean = 0, Std = 1
    
    Args:
        df (pd.DataFrame): Dataframe with numerical features
        fit_scaler (bool): Whether to fit the scaler (True for train, False for test)
        
    Returns:
        pd.DataFrame: Scaled dataframe
    """
```

##### **split_data(df: pd.DataFrame) → Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]**
Stage 5: Train-test split with stratification.

```python
def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets:
    - Stratified sampling
    - Maintains class distribution
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
```

##### **full_preprocessing_pipeline(df: pd.DataFrame) → Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]**
Execute complete preprocessing pipeline.

```python
def full_preprocessing_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute the complete 5-stage preprocessing pipeline:
    1. Data cleaning
    2. Feature engineering
    3. Categorical encoding
    4. Numerical scaling
    5. Train-test split
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test
    """
```

**Usage Example:**
```python
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(df)
print(f"Training features: {X_train.shape}")
print(f"Feature names: {preprocessor.feature_names}")
```

---

## 🤖 **model_training.py**

### **Class: ModelTrainer**
Multi-algorithm model training and evaluation system.

```python
class ModelTrainer:
    """Train and evaluate multiple ML algorithms with automated selection."""
```

#### **Attributes**
```python
config: Config                      # Configuration object
models: Dict[str, Dict]             # Model definitions
trained_models: Dict[str, Any]      # Fitted model objects
model_scores: Dict[str, Dict]       # Performance metrics
feature_importance: Dict[str, pd.DataFrame]  # Feature importance data
```

#### **Methods**

##### **initialize_models() → None**
Initialize the model portfolio.

```python
def initialize_models(self) -> None:
    """
    Initialize ML model portfolio:
    - Logistic Regression
    - Random Forest Classifier  
    - Gradient Boosting Classifier
    
    All models configured with optimal hyperparameters.
    """
```

##### **train_single_model(model_name: str, model_info: Dict, X_train: pd.DataFrame, y_train: pd.Series) → None**
Train a single model with cross-validation.

```python
def train_single_model(self, model_name: str, model_info: Dict, 
                      X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Train a single model with performance tracking:
    - Model fitting
    - Cross-validation scoring
    - Training time measurement
    
    Args:
        model_name (str): Name of the model
        model_info (Dict): Model configuration
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    """
```

##### **train_all_models(X_train: pd.DataFrame, y_train: pd.Series) → None**
Train all models in the portfolio.

```python
def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """
    Train all models in the portfolio:
    - Parallel training where possible
    - Progress tracking
    - Error handling for individual models
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    """
```

##### **evaluate_models(X_test: pd.DataFrame, y_test: pd.Series) → Dict[str, Dict]**
Evaluate all trained models on test set.

```python
def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
    """
    Comprehensive model evaluation:
    - ROC-AUC, Precision, Recall, F1-Score
    - Confusion matrices
    - ROC curves
    - Classification reports
    
    Args:
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        Dict[str, Dict]: Evaluation results for each model
    """
```

##### **compare_models(evaluation_results: Dict) → Tuple[pd.DataFrame, str]**
Compare models and select champion.

```python
def compare_models(self, evaluation_results: Dict) -> Tuple[pd.DataFrame, str]:
    """
    Compare model performance and select champion:
    - Multi-criteria evaluation
    - Automated champion selection
    - Performance ranking
    
    Args:
        evaluation_results (Dict): Results from evaluate_models()
        
    Returns:
        Tuple[pd.DataFrame, str]: Comparison dataframe and champion model name
    """
```

##### **analyze_feature_importance(model_name: str, feature_names: List[str]) → pd.DataFrame**
Analyze feature importance for interpretability.

```python
def analyze_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract and analyze feature importance:
    - Tree-based importance (for ensemble models)
    - Coefficient importance (for linear models)
    - Sorted by importance
    
    Args:
        model_name (str): Name of the model to analyze
        feature_names (List[str]): List of feature names
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
```

##### **full_training_pipeline(X_train, X_test, y_train, y_test, feature_names) → Tuple[Dict, pd.DataFrame, str]**
Execute complete training pipeline.

```python
def full_training_pipeline(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                          y_train: pd.Series, y_test: pd.Series, 
                          feature_names: List[str]) -> Tuple[Dict, pd.DataFrame, str]:
    """
    Execute complete model training pipeline:
    1. Initialize models
    2. Train all models
    3. Evaluate performance
    4. Compare and select champion
    5. Analyze feature importance
    6. Save models
    
    Args:
        X_train, X_test (pd.DataFrame): Feature sets
        y_train, y_test (pd.Series): Target sets
        feature_names (List[str]): Feature names
        
    Returns:
        Tuple[Dict, pd.DataFrame, str]: 
            evaluation_results, comparison_df, best_model_name
    """
```

---

## 🔗 **sas_connection.py**

### **Class: SASConnection**
Manages SAS environment connections and authentication.

```python
class SASConnection:
    """Handles SAS Viya connections using OAuth2 authentication."""
```

#### **Attributes**
```python
config: Config                      # Configuration object
cas_session: swat.CAS              # CAS session object
sasctl_session: Session            # sasctl session object
is_connected: bool                 # Connection status flag
```

#### **Methods**

##### **read_token_files() → Tuple[str, str]**
Read OAuth2 tokens from files.

```python
def read_token_files(self) -> Tuple[str, str]:
    """
    Read access and refresh tokens from notebook directory:
    - access_token.txt: Current access token
    - refresh_token.txt: Token refresh capability
    
    Returns:
        Tuple[str, str]: access_token, refresh_token
        
    Raises:
        FileNotFoundError: If token files don't exist
    """
```

##### **establish_full_connection() → bool**
Establish connections to both CAS and Model Manager.

```python
def establish_full_connection(self) -> bool:
    """
    Establish dual SAS connections:
    1. CAS session for analytics
    2. sasctl session for Model Manager
    
    Uses OAuth2 token authentication with SSL certificates.
    
    Returns:
        bool: True if both connections successful, False otherwise
    """
```

##### **test_cas_capabilities() → bool**
Test CAS connection with data upload/download.

```python
def test_cas_capabilities(self) -> bool:
    """
    Test CAS functionality:
    - Data upload test
    - Table operations
    - Data retrieval
    - Cleanup
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
```

##### **close_connections() → None**
Close all SAS connections.

```python
def close_connections(self) -> None:
    """
    Properly close all SAS connections:
    - Close CAS session
    - Clean up resources
    - Reset connection status
    """
```

---

## 📋 **sas_model_lifecycle.py**

### **Class: SASModelLifecycle**
Complete SAS Model Manager lifecycle management.

```python
class SASModelLifecycle:
    """Manages complete model lifecycle in SAS Model Manager."""
```

#### **Attributes**
```python
config: Config                      # Configuration object
sas_connection: SASConnection       # SAS connection manager
registered_models: Dict[str, Dict]  # Registered model metadata
best_model_name: str               # Champion model name
```

#### **Methods**

##### **prepare_model_for_sas_registration(model_name: str, X_train: pd.DataFrame, feature_names: List[str]) → bool**
Prepare model for SAS registration using PZMM.

```python
def prepare_model_for_sas_registration(self, model_name: str, X_train: pd.DataFrame, 
                                     feature_names: List[str]) -> bool:
    """
    Prepare model for SAS registration:
    - Load trained model
    - Create PZMM package
    - Generate metadata
    - Create input/output variable definitions
    
    Args:
        model_name (str): Name of the model to prepare
        X_train (pd.DataFrame): Training data for schema
        feature_names (List[str]): List of feature names
        
    Returns:
        bool: True if preparation successful, False otherwise
    """
```

##### **register_model_to_sas_manager(model_name: str, overwrite: bool = True) → bool**
Register model in SAS Model Manager.

```python
def register_model_to_sas_manager(self, model_name: str, overwrite: bool = True) → bool:
    """
    Register model in SAS Model Manager:
    - Create model object
    - Upload model files
    - Set metadata and properties
    - Configure input/output variables
    - Set champion status if applicable
    
    Args:
        model_name (str): Name of the model to register
        overwrite (bool): Whether to overwrite existing model
        
    Returns:
        bool: True if registration successful, False otherwise
    """
```

##### **score_new_data_with_cas(new_data: pd.DataFrame, model_name: str) → Dict**
Score new data using CAS.

```python
def score_new_data_with_cas(self, new_data: pd.DataFrame, model_name: str) → Dict:
    """
    Score new data using CAS:
    - Upload data to CAS
    - Apply trained model
    - Generate predictions
    - Calculate summary statistics
    
    Args:
        new_data (pd.DataFrame): New data to score
        model_name (str): Model to use for scoring
        
    Returns:
        Dict: Scoring results with predictions and statistics
    """
```

##### **generate_sas_reports(model_name: str) → bool**
Generate comprehensive SAS reports.

```python
def generate_sas_reports(self, model_name: str) → bool:
    """
    Generate SAS Model Manager reports:
    - Performance summary
    - Model metadata
    - Champion status
    - Business metrics
    
    Args:
        model_name (str): Model to generate reports for
        
    Returns:
        bool: True if report generation successful, False otherwise
    """
```

##### **register_all_models_to_sas(model_list: List[str], X_train: pd.DataFrame, feature_names: List[str], best_model_name: str, overwrite: bool = True) → Tuple[List[str], List[str]]**
Register multiple models with champion selection.

```python
def register_all_models_to_sas(self, model_list: List[str], X_train: pd.DataFrame, 
                              feature_names: List[str], best_model_name: str, 
                              overwrite: bool = True) → Tuple[List[str], List[str]]:
    """
    Register multiple models to SAS:
    - Batch model preparation
    - Sequential registration
    - Champion model designation
    - Error handling and rollback
    
    Args:
        model_list (List[str]): List of models to register
        X_train (pd.DataFrame): Training data
        feature_names (List[str]): Feature names
        best_model_name (str): Champion model name
        overwrite (bool): Whether to overwrite existing models
        
    Returns:
        Tuple[List[str], List[str]]: successful_models, failed_models
    """
```

##### **full_sas_lifecycle_pipeline(best_model_name: str, X_train: pd.DataFrame, feature_names: List[str], sample_new_data: pd.DataFrame = None, overwrite: bool = True, upload_all_models: bool = True) → bool**
Execute complete SAS lifecycle pipeline.

```python
def full_sas_lifecycle_pipeline(self, best_model_name: str, X_train: pd.DataFrame, 
                               feature_names: List[str], sample_new_data: pd.DataFrame = None, 
                               overwrite: bool = True, upload_all_models: bool = True) → bool:
    """
    Execute complete SAS lifecycle pipeline:
    1. Model registration (single or multiple)
    2. Champion selection
    3. Report generation
    4. Scoring test (optional)
    
    Args:
        best_model_name (str): Champion model name
        X_train (pd.DataFrame): Training data
        feature_names (List[str]): Feature names
        sample_new_data (pd.DataFrame, optional): Data for scoring test
        overwrite (bool): Whether to overwrite existing models
        upload_all_models (bool): Whether to upload all models or just champion
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
```

---

## 🎯 **Usage Examples**

### **Basic Usage**
```python
# Complete pipeline execution
from src.config import Config
from src.exploratory_analysis import ExploratoryAnalyzer
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.sas_connection import SASConnection
from src.sas_model_lifecycle import SASModelLifecycle

# Initialize components
config = Config()
analyzer = ExploratoryAnalyzer()
preprocessor = DataPreprocessor()
trainer = ModelTrainer()

# Load and analyze data
df = analyzer.load_data('data/raw/bank_churn.csv')
df = analyzer.basic_info(df)
df = analyzer.target_analysis(df)

# Preprocess data
X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(df)

# Train models
evaluation_results, comparison_df, best_model_name = trainer.full_training_pipeline(
    X_train, X_test, y_train, y_test, preprocessor.feature_names
)

# SAS integration
sas_conn = SASConnection()
if sas_conn.establish_full_connection():
    lifecycle = SASModelLifecycle(sas_conn)
    success = lifecycle.full_sas_lifecycle_pipeline(
        best_model_name=best_model_name,
        X_train=X_train,
        feature_names=preprocessor.feature_names
    )
    sas_conn.close_connections()
```

### **Custom Feature Engineering**
```python
# Extend DataPreprocessor for custom features
class CustomDataPreprocessor(DataPreprocessor):
    def custom_feature_engineering(self, df):
        # Add business-specific features
        df['High_Value_Customer'] = (df['Credit_Limit'] > 20000).astype(int)
        df['Transaction_Velocity'] = df['Total_Trans_Ct'] / df['Months_on_book']
        return df
    
    def full_preprocessing_pipeline(self, df):
        # Override to include custom features
        df_clean = self.clean_data(df)
        df_fe = self.feature_engineering(df_clean)
        df_custom = self.custom_feature_engineering(df_fe)  # Add custom step
        df_encoded = self.encode_categorical_features(df_custom)
        df_scaled = self.scale_numerical_features(df_encoded, fit_scaler=True)
        return self.split_data(df_scaled)
```

### **Custom Model Training**
```python
# Add custom models to the portfolio
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

class ExtendedModelTrainer(ModelTrainer):
    def initialize_models(self):
        # Call parent method
        super().initialize_models()
        
        # Add custom models
        self.models['Extra_Trees'] = {
            'model': ExtraTreesClassifier(random_state=self.config.RANDOM_STATE)
        }
        self.models['XGBoost'] = {
            'model': XGBClassifier(random_state=self.config.RANDOM_STATE)
        }
```

---

## 🚨 **Error Handling**

### **Common Exception Types**
```python
# Connection Errors
class SASConnectionError(Exception):
    """Raised when SAS connection fails"""
    pass

# Data Processing Errors  
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

# Model Training Errors
class ModelTrainingError(Exception):
    """Raised when model training fails"""
    pass

# Registration Errors
class ModelRegistrationError(Exception):
    """Raised when SAS model registration fails"""
    pass
```

### **Error Handling Patterns**
```python
# Example error handling in methods
def register_model_to_sas_manager(self, model_name: str) -> bool:
    try:
        # Model registration logic
        pass
    except SASConnectionError as e:
        logger.error(f"SAS connection failed: {e}")
        return False
    except ModelRegistrationError as e:
        logger.error(f"Model registration failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
```

---

## 📊 **Return Types and Data Structures**

### **Common Return Types**
```python
# Model evaluation results
EvaluationResults = Dict[str, Dict[str, Union[float, str, np.ndarray]]]

# Example:
{
    'Gradient_Boosting': {
        'roc_auc': 0.967,
        'precision': 0.924,
        'recall': 0.879,
        'f1_score': 0.901,
        'confusion_matrix': [[1580, 120], [89, 211]]
    }
}

# Scoring results
ScoringResults = Dict[str, Union[str, int, float]]

# Example:
{
    'table_name': 'scoring_data',
    'model_used': 'Native_Python_Gradient_Boosting',
    'records_scored': 1000,
    'mean_churn_probability': 0.156,
    'high_risk_customers': 234,
    'timestamp': '2024-01-15T14:20:00'
}
```

---

## 🔧 **Configuration Reference**

### **Complete Configuration Schema**
```python
# config.py full schema
CONFIG_SCHEMA = {
    'paths': {
        'PROJECT_ROOT': str,
        'DATA_RAW_PATH': str,
        'DATA_PROCESSED_PATH': str,
        'MODELS_PATH': str,
        'REPORTS_PATH': str
    },
    'preprocessing': {
        'TEST_SIZE': float,
        'RANDOM_STATE': int,
        'TARGET_COLUMN': str,
        'COLUMNS_TO_DROP': List[str]
    },
    'sas_config': {
        'demo_environment': str,
        'hostname': str,
        'cas_endpoint': str,
        'protocol': str,
        'verify_ssl': bool,
        'ssl_ca_list': str
    },
    'model_manager': {
        'project_name': str,
        'model_repository': str,
        'model_prefix': str
    }
}
```

---

**📚 This API reference provides complete documentation for extending and customizing the platform.** 