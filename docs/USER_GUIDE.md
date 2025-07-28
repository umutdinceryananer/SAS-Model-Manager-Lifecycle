# 📖 Complete User Guide

## 🎯 **Getting Started**

This guide will walk you through using the Bank Churn Prediction platform from basic usage to advanced customization.

---

## 🚀 **Quick Start Tutorial**

### **Your First Model in 5 Minutes**

```bash
# 1. Activate your environment
source sas_ml_env/bin/activate  # Linux/macOS
# or
sas_ml_env\Scripts\activate     # Windows

# 2. Verify setup
python test_installation.py

# 3. Run the complete pipeline
python main.py

# 4. Check results
ls reports/  # View generated reports
ls models/   # View trained models
```

### **What Happens When You Run `python main.py`**

The platform executes a 5-phase pipeline:

1. **📊 Exploratory Data Analysis**: Understand your data
2. **🔧 Data Preprocessing**: Clean and engineer features
3. **🤖 Model Training**: Train and compare 3 algorithms
4. **🔗 SAS Integration**: Connect and authenticate with SAS
5. **📋 Model Lifecycle**: Register, report, and deploy models

---

## 📊 **Understanding Your Data**

### **Dataset Overview: Bank Customer Churn**

The platform analyzes **bank customer data** to predict churn probability:

#### **Key Features**
```python
# Customer Demographics
- Customer_Age: Age of the customer
- Gender: M/F
- Dependent_count: Number of dependents
- Education_Level: Education background
- Marital_Status: Single/Married/Divorced

# Financial Profile  
- Income_Category: Annual income bracket
- Credit_Limit: Available credit limit
- Avg_Open_To_Buy: Available purchasing power
- Total_Amt_Chng_Q4_Q1: Amount change Q4 to Q1

# Behavioral Patterns
- Total_Trans_Amt: Total transaction amount
- Total_Trans_Ct: Total transaction count
- Avg_Utilization_Ratio: Credit utilization
- Contacts_Count_12_mon: Contact frequency
- Months_Inactive_12_mon: Inactive months

# Target Variable
- Attrition_Flag: "Existing Customer" or "Attrited Customer"
```

#### **Data Quality Insights**
```python
# Dataset Statistics (typical)
Total Records: ~10,000 customers
Churn Rate: ~16% (realistic business ratio)
Missing Values: Minimal (<2% in most columns)
Feature Types: Mixed (numerical, categorical, ordinal)
```

---

## 🔧 **Data Preprocessing Deep Dive**

### **5-Stage Preprocessing Pipeline**

#### **Stage 1: Data Cleaning**
```python
# What happens:
# ✅ Remove unnecessary columns (CLIENTNUM, Naive_Bayes_*)  
# ✅ Convert target to binary (Churn: 0/1)
# ✅ Handle multicollinearity (remove Avg_Open_To_Buy)
# ✅ Validate data types and ranges

# Why it matters:
# - Removes noise and redundant information
# - Creates consistent target variable
# - Prevents model overfitting
```

#### **Stage 2: Feature Engineering**
```python
# Advanced features created:
Age_Group = pd.cut(Customer_Age, bins=[0, 35, 50, 65, 100], 
                   labels=['Young', 'Middle', 'Senior', 'Elder'])

Utilization_Category = pd.cut(Avg_Utilization_Ratio,
                             bins=[0, 0.3, 0.7, 1.0],
                             labels=['Low', 'Medium', 'High'])

# Interaction features:
Income_Credit_Ratio = Income_Category / Credit_Limit
Transaction_Frequency = Total_Trans_Ct / 12  # Monthly average
Risk_Score = Weighted combination of risk indicators

# Business value:
# - Captures non-linear relationships
# - Creates interpretable business metrics
# - Improves model performance
```

#### **Stage 3: Categorical Encoding**
```python
# Encoding strategies:
# 📋 Label Encoding: Ordinal categories (Education_Level)
# 🔢 Target Encoding: High-cardinality categories
# 📊 One-hot Encoding: Low-cardinality nominal features

# Example:
Education_Level: 
  'High School' → 1, 'Graduate' → 2, 'Post-Graduate' → 3
Gender: 
  'M' → [1,0], 'F' → [0,1]
```

#### **Stage 4: Feature Scaling**
```python
# StandardScaler applied to numerical features:
scaled_feature = (feature - mean) / std_deviation

# Benefits:
# ✅ Ensures equal weight for all features
# ✅ Improves convergence for gradient-based algorithms
# ✅ Required for logistic regression
```

#### **Stage 5: Train-Test Split**
```python
# Stratified split maintains class distribution:
X_train: 80% of data (8,000 records)
X_test:  20% of data (2,000 records)
Churn rate maintained: ~16% in both sets

# Why stratified:
# - Prevents data leakage
# - Ensures representative evaluation
# - Maintains business reality in test set
```

---

## 🤖 **Model Training & Selection**

### **Multi-Algorithm Approach**

#### **Algorithm Portfolio**
```python
1. 📊 Logistic Regression
   ✅ Fast training and prediction
   ✅ Highly interpretable coefficients
   ✅ Good baseline performance
   ⚠️  Assumes linear relationships

2. 🌲 Random Forest  
   ✅ Handles non-linear patterns
   ✅ Built-in feature importance
   ✅ Robust to outliers
   ⚠️  Can overfit with small datasets

3. 🚀 Gradient Boosting
   ✅ Often best performance
   ✅ Handles complex interactions
   ✅ Excellent for tabular data
   ⚠️  Longer training time
```

#### **Champion Selection Process**
```python
# Evaluation metrics (in order of importance):
1. ROC-AUC Score: Area under ROC curve
2. Cross-Validation Stability: 5-fold CV std deviation
3. Precision: Minimizing false positives
4. Recall: Catching actual churners
5. F1-Score: Balanced precision/recall

# Champion selection logic:
best_model = max(models, key=lambda m: m.roc_auc_score)
if cv_stability < threshold:
    best_model = most_stable_model
```

### **Model Interpretation**

#### **Feature Importance Analysis**
```python
# Top churn predictors (typical results):
1. Total_Trans_Amt: -0.34      # Higher transactions = lower churn
2. Total_Trans_Ct: -0.28       # More frequent usage = retention  
3. Avg_Utilization_Ratio: 0.25 # High utilization = risk
4. Contacts_Count_12_mon: 0.22  # More contacts = likely to churn
5. Months_Inactive_12_mon: 0.19 # Inactivity predicts churn

# Business insights:
# 💡 Encourage transaction frequency
# 💡 Monitor high utilization customers  
# 💡 Proactive engagement reduces churn
```

#### **Model Performance Interpretation**
```python
# Typical champion model results:
ROC-AUC: 0.967   # Excellent discrimination
Precision: 0.924 # 92% of predicted churners actually churn
Recall: 0.879    # Catches 88% of actual churners
F1-Score: 0.901  # Excellent balance

# Business translation:
# 📈 Can identify 88% of potential churners
# 💰 Only 8% false alarms (efficient targeting)
# 🎯 ROI: $2.3M annual savings (example)
```

---

## 🔗 **SAS Integration Guide**

### **Phase 1: Authentication & Connection**

#### **OAuth2 Token Management**
```python
# Token lifecycle:
1. Initial generation (follow TOKEN_SETUP.md)
2. Automatic refresh (handled by platform)
3. Secure storage (local files, excluded from Git)
4. Session management (automatic)

# Token validation:
def validate_token():
    response = requests.get(
        "https://create.demo.sas.com/identities/users/@currentUser",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    return response.status_code == 200
```

#### **Dual Connection Architecture**
```python
# CAS Connection (Analytics)
cas_session = swat.CAS(
    "https://create.demo.sas.com/cas-shared-default-http",
    username=None,
    password=access_token,
    protocol="https"
)

# sasctl Session (Model Manager)
sasctl_session = Session(
    hostname="https://create.demo.sas.com",
    token=access_token,
    verify_ssl=False
)
```

### **Phase 2: Model Registration**

#### **PZMM Model Packaging**
```python
# What PZMM creates:
models/pzmm_packages/
├── Native_Python_Gradient_Boosting/
│   ├── metadata.json           # Model metadata
│   ├── inputVar.json          # Input variable definitions
│   ├── outputVar.json         # Output variable definitions
│   ├── ModelProperties.json   # SAS model properties  
│   ├── score.py              # Scoring code
│   └── model.pickle          # Serialized model

# Automatic metadata generation:
{
  "name": "Native_Python_Gradient_Boosting",
  "description": "Bank customer churn prediction using Gradient_Boosting",
  "algorithm": "GradientBoostingClassifier",
  "target_variable": "Churn",
  "input_variables": ["Customer_Age", "Credit_Limit", ...],
  "model_type": "Classification",
  "creation_date": "2024-01-15T10:30:00"
}
```

#### **Champion Model Promotion**
```python
# Automatic champion designation:
for model in trained_models:
    is_champion = (model == best_model_name)
    register_model_to_sas(
        model_name=model,
        is_champion=is_champion,
        project="Bank_Churn_Prediction"
    )

# SAS Model Manager integration:
# ✅ Model registered with metadata
# ✅ Champion status set automatically
# ✅ Input/output variables defined
# ✅ Scoring code generated
```

### **Phase 3: Real-time Scoring**

#### **CAS Scoring Engine**
```python
# Upload new data for scoring:
new_customers = pd.read_csv("new_customers.csv")
scoring_results = lifecycle.score_new_data_with_cas(
    new_data=new_customers,
    model_name="Gradient_Boosting"
)

# Scoring output:
{
  'records_scored': 1000,
  'mean_churn_probability': 0.156,
  'high_risk_customers': 234,  # Probability > 0.5
  'timestamp': '2024-01-15T14:20:00'
}
```

---

## 📊 **Reports & Analytics**

### **Generated Reports**

#### **Model Performance Report**
```
📁 reports/model_performance/
├── model_comparison.json      # Algorithm comparison
├── champion_analysis.json     # Best model details
├── feature_importance.json    # Variable impact analysis
└── business_metrics.json      # ROI and impact calculations
```

#### **SAS Model Manager Reports**  
```
📁 reports/sas_reports/
├── Gradient_Boosting_sas_report.json
├── Random_Forest_sas_report.json
└── Logistic_Regression_sas_report.json

# Report content:
{
  "model_name": "Native_Python_Gradient_Boosting",
  "status": "Champion",
  "algorithm": "GradientBoostingClassifier", 
  "features_count": 25,
  "registration_time": "2024-01-15T10:30:00",
  "is_champion": true
}
```

#### **Example Business Metrics**
```python
# Example metrics that could be tracked:
Model Performance: ROC-AUC 96.7%, Precision 92.4%
Prediction Coverage: 1000 customers scored daily
Campaign Targeting: 234 high-risk customers identified
Cost Efficiency: Reduced false positive rate to 8%

# ROI Calculation Framework:
# Investment: Model development + deployment costs
# Returns: Value of retained customers minus campaign costs
# Actual ROI will depend on specific business context
```

---

## 🎛️ **Customization Guide**

### **Configuration Options**

#### **Model Hyperparameters**
```python
# src/config.py customization:
MODEL_HYPERPARAMETERS = {
    'Gradient_Boosting': {
        'n_estimators': [50, 100, 200],      # More estimators = better performance
        'learning_rate': [0.05, 0.1, 0.2],  # Lower = more stable
        'max_depth': [3, 5, 7],              # Higher = more complex
        'min_samples_split': [2, 5, 10]      # Higher = less overfitting  
    }
}

# Grid search automatically finds optimal combination
```

#### **Feature Engineering Options**
```python
# Customize feature creation:
FEATURE_ENGINEERING_CONFIG = {
    'age_bins': [18, 30, 45, 60, 100],        # Age group boundaries
    'utilization_thresholds': [0.3, 0.7],     # Low/Medium/High utilization
    'create_interaction_features': True,       # Feature combinations
    'polynomial_features': False,              # Polynomial transformations
    'target_encoding': True                    # High-cardinality encoding
}
```

#### **Business Rules Integration**
```python
# Add domain-specific logic:
def apply_business_rules(predictions, customer_data):
    # Rule 1: VIP customers get special handling
    vip_mask = customer_data['Credit_Limit'] > 50000
    predictions[vip_mask] *= 1.2  # Higher sensitivity
    
    # Rule 2: Recent customers are less likely to churn
    recent_mask = customer_data['Months_on_book'] < 12
    predictions[recent_mask] *= 0.8  # Lower churn probability
    
    return predictions
```

### **Advanced Customizations**

#### **Custom Evaluation Metrics**
```python
# Add business-specific metrics:
def customer_lifetime_value_score(y_true, y_pred, customer_values):
    """Calculate CLV-weighted model performance."""
    weighted_correct = np.sum(
        (y_true == y_pred) * customer_values
    )
    total_value = np.sum(customer_values)
    return weighted_correct / total_value

# Integrate into model selection:
def select_champion_model_clv(models, X_test, y_test, customer_values):
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        clv_score = customer_lifetime_value_score(
            y_test, y_pred, customer_values
        )
        if clv_score > best_score:
            best_score = clv_score
            best_model = name
    
    return best_model, best_score
```

#### **Custom Feature Engineering**
```python
# Industry-specific features:
def create_banking_features(df):
    # Credit behavior patterns
    df['Credit_Utilization_Trend'] = (
        df['Avg_Utilization_Ratio'] - df['Avg_Utilization_Ratio'].rolling(3).mean()
    )
    
    # Transaction patterns
    df['Transaction_Consistency'] = df['Total_Trans_Ct'] / df['Total_Trans_Ct'].std()
    
    # Risk indicators
    df['Payment_Behavior_Score'] = (
        df['Total_Revolving_Bal'] / df['Credit_Limit'] * 
        df['Avg_Utilization_Ratio']
    )
    
    # Engagement metrics
    df['Digital_Engagement'] = (
        df['Total_Trans_Ct'] * df['Total_Trans_Amt'] / 
        df['Months_on_book']
    )
    
    return df
```

---

## 🚨 **Troubleshooting Common Issues**

### **Data Issues**

#### **Problem: Poor Model Performance**
```python
# Diagnosis steps:
1. Check data quality:
   - Missing values > 10%?
   - Class imbalance > 90/10?
   - Feature scaling issues?

2. Validate features:
   - Are features predictive individually?
   - Check feature correlation matrix
   - Look for data leakage

3. Model diagnostics:
   - Cross-validation scores consistent?
   - Learning curves show overfitting?
   - Try different algorithms

# Solutions:
# ✅ Increase data quality thresholds
# ✅ Apply class balancing techniques  
# ✅ Use feature selection methods
# ✅ Tune hyperparameters more aggressively
```

#### **Problem: SAS Integration Failures**
```python
# Common causes and solutions:

1. Token Expiration:
   # Check token validity
   # Regenerate tokens using TOKEN_SETUP.md guide
   # Verify token file permissions

2. Network Connectivity:
   # Test: ping create.demo.sas.com
   # Check firewall settings
   # Verify SSL certificates

3. Model Registration Issues:
   # Verify project exists in Model Manager
   # Check user permissions
   # Validate model artifacts in pzmm_packages/

4. Scoring Failures:
   # Ensure data schema matches training
   # Check feature names and types
   # Validate CAS table uploads
```

### **Performance Issues**

#### **Problem: Slow Training**
```python
# Optimization strategies:
1. Data sampling for development:
   DEVELOPMENT_SAMPLE_SIZE = 5000
   
2. Parallel processing:
   n_jobs = -1  # Use all CPU cores
   
3. Feature selection:
   # Use only top N features
   # Remove highly correlated features
   
4. Algorithm selection:
   # Start with faster algorithms
   # Use early stopping for boosting
```

#### **Problem: Memory Errors**
```python
# Memory optimization:
1. Reduce data types:
   df = df.astype({'feature': 'float32'})  # vs float64
   
2. Process in chunks:
   for chunk in pd.read_csv('large_file.csv', chunksize=1000):
       process_chunk(chunk)
       
3. Free memory explicitly:
   del large_dataframe
   gc.collect()
```

---

## 📚 **Best Practices**

### **Development Workflow**
```bash
# Recommended development cycle:
1. Start with small data sample (1000 records)
2. Validate preprocessing pipeline
3. Test single model training
4. Verify SAS connection
5. Scale to full dataset
6. Run complete pipeline
7. Validate results in SAS Model Manager
```

### **Production Deployment**
```python
# Production checklist:
✅ Data validation pipeline
✅ Model performance monitoring
✅ Automated retraining schedule
✅ A/B testing framework
✅ Rollback procedures
✅ Security compliance
✅ Documentation and runbooks
```

### **Monitoring & Maintenance**
```python
# Key metrics to monitor:
1. Model Performance:
   - AUC score drift
   - Prediction distribution changes
   - False positive/negative rates

2. Data Quality:
   - Missing value increases
   - Feature distribution shifts
   - New categorical values

3. Business Impact:
   - Actual churn rate changes
   - Campaign response rates
   - Revenue impact tracking
```

---

## 🎓 **Advanced Topics**

### **Model Ensemble Techniques**
```python
# Combine multiple models for better performance:
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier([
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()), 
    ('gb', GradientBoostingClassifier())
], voting='soft')

# Benefits:
# ✅ Often better than single models
# ✅ More robust predictions
# ✅ Captures different patterns
```

### **Online Learning for Model Updates**
```python
# Incremental learning for production:
from sklearn.linear_model import SGDClassifier

online_model = SGDClassifier(loss='log')

# Update with new data:
def update_model_online(new_X, new_y):
    online_model.partial_fit(new_X, new_y)
    
# Benefits:
# ✅ Fast updates with new data
# ✅ No need to retrain from scratch
# ✅ Adapts to changing patterns
```

### **Explainable AI Integration**
```python
# Add model interpretability:
import shap

# Generate SHAP explanations:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize feature importance:
shap.summary_plot(shap_values, X_test)

# Per-prediction explanations:
shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

---

## 📞 **Support & Resources**

### **Getting Help**
1. **📖 Check this User Guide** for common scenarios
2. **🔍 Review Troubleshooting section** for known issues  
3. **📧 Contact Support** with detailed logs and error messages
4. **💬 Join Community Forums** for discussions and tips

### **Additional Resources**
- **SAS Viya Documentation**: [https://documentation.sas.com](https://documentation.sas.com)
- **Scikit-learn User Guide**: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **Python SWAT Documentation**: [https://sassoftware.github.io/python-swat/](https://sassoftware.github.io/python-swat/)
- **SAS Model Manager**: [SAS Model Manager Documentation](https://documentation.sas.com/?cdcId=mdlmgrcdc)

---

**🎉 You're now ready to master the Bank Churn Prediction platform!** 