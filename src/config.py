import os

class Config:
    # Proje yolları
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
    DATA_PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
    MODELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained')
    REPORTS_PATH = os.path.join(PROJECT_ROOT, 'reports')
    
    # Model parametreleri
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    TARGET_COLUMN = 'Attrition_Flag'
    
    # Kaldırılacak sütunlar
    COLUMNS_TO_DROP = [
        'CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
    
    SAS_CONFIG = {
        # Demo Environment Options
        'demo_environment': 'Create', 
        'environment_urls': {
            'Create': {
                'hostname': 'Create.demo.sas.com', 
                'ip_address': '4.236.232.19',
                'cas_endpoint': 'https://Create.demo.sas.com/cas-shared-default-http'
            }
        },
        
        # Authentication (Token-based)
        'client_id': 'api.client',
        'access_token': None,  # Runtime'da generate edilecek
        'refresh_token': None,
        'token_expiry': None,
        
        # Connection Settings
        'connection_type': 'http',  # 'http' veya 'binary'
        'protocol': 'https',
        'verify_ssl': True,  # Demo environment için True
        
        # Certificate Paths (kullanıcı indirmeli)
        'ssl_ca_list': os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'certificates', 
            'demo-rootCA-Intermidiates_4CLI.pem'
        ),
        'binary_cert': None,  # ssemonthly-bin.crt path
        
        # Legacy (artık kullanılmıyor)
        'username': None,  # Token authentication'da gerekmiyor
        'password': None   # Token authentication'da gerekmiyor
    }
    
    # SAS Model Manager Settings
    MODEL_MANAGER_CONFIG = {
        'project_name': 'Bank_Churn_Prediction',
        'model_repository': 'Public',
        'model_prefix': 'Native_Python'
    }

    