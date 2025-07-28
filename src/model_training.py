import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from .config import Config

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.feature_importance = {}
        
    def initialize_models(self):
        print("MODEL PORTFOLIO:")

        self.models = {
            'Logistic_Regression': {
                'model': LogisticRegression(random_state=self.config.RANDOM_STATE, max_iter=1000)
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=self.config.RANDOM_STATE, n_estimators=100)
            },
            'Gradient_Boosting': {
                'model': GradientBoostingClassifier(random_state=self.config.RANDOM_STATE, n_estimators=100)
            }
        }
        
        for name, info in self.models.items():
            print(f"{name}")
                
    def train_single_model(self, model_name, model_info, X_train, y_train):
        print(f"\nTraining {model_name}")
        
        start_time = time.time()
        
        model = model_info['model']
        model.fit(X_train, y_train)
        
        # Cross-Validation Score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        training_time = time.time() - start_time
        
        self.trained_models[model_name] = model
        self.model_scores[model_name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
        
        print(f"CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"Training Time: {training_time:.2f} Seconds")
        
        return model
    
    def train_all_models(self, X_train, y_train):
        print("\n" + "=" * 60)
        print("PHASE 3.1: MODEL TRAINING")
        print("=" * 60)
        
        for model_name, model_info in self.models.items():
            self.train_single_model(model_name, model_info, X_train, y_train)
            
        print(f"\nAll Models trained successfully!")
        
    def evaluate_models(self, X_test, y_test):
        print("\n" + "=" * 60)
        print("PHASE 3.2: MODEL EVALUATION")
        print("=" * 60)
        
        evaluation_results = {}
        
        for model_name, model in self.trained_models.items():
            print(f"\n{model_name} Results:")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            evaluation_results[model_name] = {
                'metrics': metrics,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1']:.4f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return evaluation_results
    
    def compare_models(self, evaluation_results):
        print("\n" + "=" * 60)
        print("PHASE 3.3: MODEL COMPARISON")
        print("=" * 60)
        
        # Results DataFrame
        comparison_data = []
        for model_name, results in evaluation_results.items():
            row = {'Model': model_name}
            row.update(results['metrics'])
            row['CV_ROC_AUC'] = self.model_scores[model_name]['cv_mean']
            row['Training_Time'] = self.model_scores[model_name]['training_time']
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        print("\nModel Performance Table:")
        print(comparison_df.to_string(index=False))
        
        # Choose Best Model (According to ROC-AUC)
        best_model_name = comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'Model']
        best_score = comparison_df.loc[comparison_df['roc_auc'].idxmax(), 'roc_auc']
        
        print(f"\nBest Model is {best_model_name} and its ROC-AUC Score is {best_score:.4f}")
        return comparison_df, best_model_name
    
    def analyze_feature_importance(self, model_name, feature_names):
        if model_name not in self.trained_models:
            print(f"{model_name} cant be found in trained models!")
            return
            
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0]) 
        else:
            print(f"{model_name} does not support feature importance analysis!")
            return
        
        # Feature importance DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        
        return feature_imp_df
    
    def save_models(self):
        print("\n" + "=" * 50)
        print(f"PHASE 3.4: SAVING MODELS")
        print("=" * 50)
        
        os.makedirs(self.config.MODELS_PATH, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
    
            model_path = os.path.join(self.config.MODELS_PATH, f"{model_name}.joblib")
            
            joblib.dump(model, model_path)
            print(f"{model_name} Saved: {model_path}")
        
        scores_path = os.path.join(self.config.MODELS_PATH, "model_scores.joblib")
        joblib.dump(self.model_scores, scores_path)
        print(f"Model Scores Saved: {scores_path}")
        
        print(f"\nAll Models are Saved into: {self.config.MODELS_PATH}")
    
    def full_training_pipeline(self, X_train, X_test, y_train, y_test, feature_names):
        
        self.initialize_models()
        
        self.train_all_models(X_train, y_train)
        
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        comparison_df, best_model_name = self.compare_models(evaluation_results)
        
        self.analyze_feature_importance(best_model_name, feature_names)
        
        self.save_models()

        print("\n" + "=" * 70)
        print("MODEL TRAINING COMPLETED")
        print("=" * 70)
        
        return evaluation_results, comparison_df, best_model_name