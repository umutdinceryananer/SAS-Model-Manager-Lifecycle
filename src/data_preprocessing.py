import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from .config import Config

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def clean_data(self, df):
        """Data Cleaning and Preparation"""
        
        # 1. Create Copy
        df_clean = df.copy()
        
        # 2. Remove Unnecessary Columns
        columns_to_drop = self.config.COLUMNS_TO_DROP
        existing_cols = [col for col in columns_to_drop if col in df_clean.columns]
        
        if existing_cols:
            df_clean = df_clean.drop(columns=existing_cols)
            print(f"✓ Removed Columns: {existing_cols}")
        
        # 3. Convert Target Variable to Binary
        if self.config.TARGET_COLUMN in df_clean.columns:
            df_clean['Churn'] = (df_clean[self.config.TARGET_COLUMN] == 'Attrited Customer').astype(int)
            df_clean = df_clean.drop(columns=[self.config.TARGET_COLUMN])
            print(f"✓ Target Variable '{self.config.TARGET_COLUMN}' Converted to Binary (Churn: 1/0)")
        
        # 4. Solve Multicollinearity Issue
        # Correlation between 'Credit_Limit' and 'Avg_Open_To_Buy'
        if 'Avg_Open_To_Buy' in df_clean.columns:
            df_clean = df_clean.drop(columns=['Avg_Open_To_Buy'])
            print("✓ Multicollinearity: Avg_Open_To_Buy Removed")
        
        return df_clean
    
    def feature_engineering(self, df):
        """Feature Engineering"""
        print("\n" + "=" * 50)
        print("PHASE 2.1: FEATURE ENGINEERING")
        print("=" * 50)
        
        df_fe = df.copy()
        
        # 1. Age Grouping
        df_fe['Age_Group'] = pd.cut(df_fe['Customer_Age'], 
                                   bins=[0, 35, 50, 65, 100], 
                                   labels=['Young', 'Middle', 'Senior', 'Elder'])
        print("✓ Age_Group Created (Young/Middle/Senior/Elder)")
        
        # 2. Credit Utilization Category
        df_fe['Utilization_Category'] = pd.cut(df_fe['Avg_Utilization_Ratio'],
                                              bins=[0, 0.3, 0.7, 1.0],
                                              labels=['Low', 'Medium', 'High'])
        print("✓ Utilization_Category Created (Low/Medium/High)")
        
        # 3. Monthly Transaction Density
        df_fe['Monthly_Trans_Ct'] = df_fe['Total_Trans_Ct'] / 12
        df_fe['Monthly_Trans_Amt'] = df_fe['Total_Trans_Amt'] / 12
        print("✓ Monthly Transaction Density Created")
        
        # 4. Customer Value Score
        df_fe['Customer_Value_Score'] = (
            df_fe['Total_Trans_Amt'] * 0.4 +  # Monetary value
            df_fe['Total_Trans_Ct'] * 0.3 +   # Frequency  
            df_fe['Total_Relationship_Count'] * 0.3  # Depth
        )
        print("✓ Customer_Value_Score Created")
        
        # 5. Risk Score
        df_fe['Risk_Score'] = (
            df_fe['Months_Inactive_12_mon'] * 2 + 
            df_fe['Contacts_Count_12_mon']
        )
        print("✓ Risk_Score Created")
        
        # 6. Credit to Income Ratio
        income_mapping = {
            'Less than $40K': 30000,
            '$40K - $60K': 50000, 
            '$60K - $80K': 70000,
            '$80K - $120K': 100000,
            '$120K +': 150000,
            'Unknown': df_fe['Credit_Limit'].median()  # Average income for unknowns
        }
        
        df_fe['Est_Income'] = df_fe['Income_Category'].map(income_mapping)
        df_fe['Credit_to_Income_Ratio'] = df_fe['Credit_Limit'] / df_fe['Est_Income']
        print("✓ Credit_to_Income_Ratio Created")
        
        print(f"✓ Size After Feature engineering: {df_fe.shape}")
        return df_fe
    
    def encode_categorical_features(self, df):
        """Kategorik değişkenleri encode et"""
        print("\n" + "=" * 50)
        print("PHASE 2.2: CATEGORICAL ENCODING")
        print("=" * 50)
        
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical Columns: {categorical_cols}")
        
        binary_cols = []
        for col in categorical_cols:
            if df_encoded[col].nunique() == 2:
                binary_cols.append(col)
        
        if binary_cols:
            for col in binary_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
                print(f"✓ {col}: Label Encoding Applied")
        
        multi_categorical = [col for col in categorical_cols if col not in binary_cols]
        
        if multi_categorical:
            df_encoded = pd.get_dummies(df_encoded, 
                                       columns=multi_categorical, 
                                       prefix=multi_categorical,
                                       drop_first=True)  # İlk kategoriyi drop et (dummy trap)
            print(f"✓ One-Hot Encoding Applied: {multi_categorical}")
        
        print(f"✓ Size After Encoding: {df_encoded.shape}")
        return df_encoded
    
    def scale_numerical_features(self, df, fit_scaler=True):
        print("\n" + "=" * 50)
        print("PHASE 2.3: NUMERICAL SCALING")
        print("=" * 50)
        
        df_scaled = df.copy()
        
        if 'Churn' in df_scaled.columns:
            target = df_scaled['Churn']
            df_features = df_scaled.drop(columns=['Churn'])
        else:
            target = None
            df_features = df_scaled
 
        numerical_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit_scaler:
            df_features[numerical_cols] = self.scaler.fit_transform(df_features[numerical_cols])
            print(f"✓ Fit Scaler and {len(numerical_cols)} columns scaled")
        else:
            df_features[numerical_cols] = self.scaler.transform(df_features[numerical_cols])
        
        if target is not None:
            df_final = df_features.copy()
            df_final['Churn'] = target
        else:
            df_final = df_features
        
        self.feature_names = df_features.columns.tolist()
        
        print(f"✓ Final Size: {df_final.shape}")
        return df_final
    
    def split_data(self, df):
        """Veriyi train-test olarak böl"""
        print("\n" + "=" * 50)
        print(f"PHASE 2.4: DATA SPLITTING")
        print("=" * 50)
        
        if 'Churn' not in df.columns:
            raise ValueError("Cant find Target Variable")
        
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        print(f"✓ Training Set: {X_train.shape[0]} Examples")
        print(f"✓ Test set: {X_test.shape[0]} Examples")
        print(f"✓ Feature Count: {X_train.shape[1]}")
        print(f"✓ Train Churn Ratio: {y_train.mean():.3f}")
        print(f"✓ Test Churn Ratio: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def full_preprocessing_pipeline(self, df):
        """Tam preprocessing pipeline'ı"""
        
        # 1. Data cleaning
        df_clean = self.clean_data(df)
        
        # 2. Feature engineering
        df_fe = self.feature_engineering(df_clean)
        
        # 3. Categorical encoding
        df_encoded = self.encode_categorical_features(df_fe)
        
        # 4. Numerical scaling
        df_scaled = self.scale_numerical_features(df_encoded, fit_scaler=True)
        
        # 5. Train-test split
        X_train, X_test, y_train, y_test = self.split_data(df_scaled)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test