# src/exploratory_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .config import Config

class ExploratoryAnalyzer:
    def __init__(self):
        self.config = Config()
        
    def load_data(self, file_path):
        """Load Data"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data Loaded Successfuly.\n")
            return df
        except Exception as e:
            print(f"Data Load Error: {e}")
            return None
    
    def basic_info(self, df):
        """Basic Information about Data"""
        print("=" * 50)
        print("PHASE 1.1: BASIC INFORMATION")
        print("=" * 50)
        
        print(f"Data Size: {df.shape[0]} rows and {df.shape[1]} columns")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

        return df
    
    def target_analysis(self, df):
        """Target Variable Analysis"""
        if self.config.TARGET_COLUMN not in df.columns:
            print(f"Hedef değişken '{self.config.TARGET_COLUMN}' bulunamadı!")
            print(f"Mevcut sütunlar: {list(df.columns)}")
            return df
            
        print("=" * 50)
        print("PHASE 1.2: TARGET VARIABLE ANALYSIS (CHURN)")
        print("=" * 50)
        
        target_counts = df[self.config.TARGET_COLUMN].value_counts()
        print(f"Churn Distribution:")
        
        minority_ratio = min(target_counts) / len(df)
        if minority_ratio < 0.1:
            print(f"Serious Imbalance. Minority class: {minority_ratio*100:.2f}%")
        elif minority_ratio < 0.3:
            print(f"Little Imbalance. Minority class: {minority_ratio*100:.2f}%")
        else:
            print(f"Balanced. Minority class: {minority_ratio*100:.2f}%")
        
        return df