import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.exploratory_analysis import ExploratoryAnalyzer
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.sas_connection import SASConnection
from src.sas_model_lifecycle import SASModelLifecycle
from src.config import Config
import pandas as pd

def main():
    config = Config()
    analyzer = ExploratoryAnalyzer()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    # https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
    data_file = os.path.join(config.DATA_RAW_PATH, 'bank_churn.csv')
 
    print("BANK CHURN PREDICTION PROJECT WITH SAS MODEL MANAGER INTEGRATION")
 
    if not os.path.exists(data_file):
        print(f"Can't Find Dataset: {data_file}")
        return

    print("\n" + "="*60)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    df = analyzer.load_data(data_file)
    if df is None:
        return
    
    df = analyzer.basic_info(df)
    df = analyzer.target_analysis(df)

    print("\n" + "="*60)
    print("PHASE 2: DATA PREPROCESSING")
    print("="*60)
    
    try:
        X_train, X_test, y_train, y_test = preprocessor.full_preprocessing_pipeline(df)
    except Exception as e:
        print(f"Preprocessing Error: {e}")
        return

    print("\n" + "="*60)
    print("PHASE 3: MODEL TRAINING & EVALUATION")
    print("="*60)
    
    try:
        evaluation_results, comparison_df, best_model_name = trainer.full_training_pipeline(
            X_train, X_test, y_train, y_test, preprocessor.feature_names
        )
    except Exception as e:
        print(f"Model Training Error: {e}")
        return

    print("\n" + "="*60)
    print("PHASE 4: SAS CONNECTION & AUTHENTICATION")
    print("="*60)
    
    sas_conn = SASConnection()
    
    connection_success = sas_conn.establish_full_connection()
    
    if not connection_success:
        print("\nSAS bağlantısı kurulamadı!")
        return
    
    sas_conn.test_cas_capabilities()

    print("\n" + "="*60)
    print("PHASE 5: SAS MODEL LIFECYCLE MANAGEMENT")
    print("="*60)
    
    sas_lifecycle = SASModelLifecycle(sas_conn)
    
    try:
        sample_scoring_data = X_test.head(50).copy()
        
        lifecycle_success = sas_lifecycle.full_sas_lifecycle_pipeline(
            best_model_name=best_model_name,
            X_train=X_train,
            feature_names=preprocessor.feature_names,
            sample_new_data=sample_scoring_data,
            upload_all_models=True
)
        
        if lifecycle_success:
            print("\nSAS MODEL LIFECYCLE BAŞARIYLA TAMAMLANDI!")
        else:
            print("\nSAS Model Lifecycle kısmen tamamlandı")
            
    except Exception as e:
        print(f"SAS Lifecycle hatası: {e}")
    
    finally:
        sas_conn.close_connections()

    print("\n" + "="*70)
    print("PROJECT COMPLETION SUMMARY")
    print("="*70)
    print(f"Data Processing: {X_train.shape[0]} training samples")
    print(f"Feature Engineering: {len(preprocessor.feature_names)} features")
    print(f"Model Training: {len(trainer.trained_models)} models trained")
    print(f"Best Model: {best_model_name}")
    
    if connection_success:
        print(f"SAS Integration: python-swat & python-sasctl kullanıldı")
        print(f"Model Lifecycle: Registration, Reports, Scoring")
    else:
        print(f"SAS Integration: Bağlantı sorunları yaşandı")

    
    print(f"\nTüm çıktılar kaydedildi:")
    print(f"• Models: {config.MODELS_PATH}")
    print(f"• Data: {config.DATA_PROCESSED_PATH}")
    print(f"• Reports: {config.REPORTS_PATH}")


if __name__ == "__main__":
    main()