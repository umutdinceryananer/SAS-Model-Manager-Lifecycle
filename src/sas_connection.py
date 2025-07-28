import swat
from sasctl import Session
import os

try:
    from .config import Config
except ImportError:
    import config
    Config = config.Config

class SASConnection:
    def __init__(self):
        self.config = Config()
        self.cas_session = None
        self.sasctl_session = None
        self.is_connected = False
        
    def read_token_files(self):
        print("Reading token files...")
        
        access_token_path = "notebooks/access_token.txt"
        refresh_token_path = "notebooks/refresh_token.txt"
        
        try:
            # Read access token
            if os.path.exists(access_token_path):
                with open(access_token_path, 'r') as f:
                    access_token = f.read().strip()
                print(f"Access token loaded from {access_token_path}")
            else:
                print(f"{access_token_path} not found!")
                return None, None
            
            # Read refresh token (optional)
            refresh_token = None
            if os.path.exists(refresh_token_path):
                with open(refresh_token_path, 'r') as f:
                    refresh_token = f.read().strip()
                print(f"Refresh token loaded from {refresh_token_path}")
            else:
                print(f"{refresh_token_path} not found (optional)")
            
            return access_token, refresh_token
            
        except Exception as e:
            print(f"Error reading token files: {e}")
            return None, None
    
    def establish_full_connection(self):
        
        env = "Create"
        print(f"Environment: {env}")
        
        access_token, refresh_token = self.read_token_files()
        
        if not access_token:
            print("No valid access token found!")
            return False

        cert_path = self.config.SAS_CONFIG.get('ssl_ca_list')
        if not cert_path or not os.path.exists(cert_path):
            cert_path = input("Certificate path (.pem): ").strip()
        
        try:
            # 1. CAS Connection
            print(f"Connecting to {env} CAS...")
            self.cas_session = swat.CAS(
                f"https://{env}.demo.sas.com/cas-shared-default-http",
                username=None,
                password=access_token,
                ssl_ca_list=cert_path,
                protocol="https"
            )
            print("CAS connection successful!")
            
            # 2. sasctl Session
            print(f"Connecting to {env} Viya for Model Manager...")
            self.sasctl_session = Session(
                hostname=f"https://{env}.demo.sas.com",
                token=access_token,
                verify_ssl=False
            )
            print("sasctl session successful!")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def test_cas_capabilities(self):
        """Test CAS capabilities"""
        if not self.cas_session:
            print("No CAS Connection established!")
            return False

        print("\n" + "=" * 50)    
        print("PHASE 4.1: CAS Capabilities Test")
        print("=" * 50)
        
        try:
            import pandas as pd
            test_data = pd.DataFrame({
                'x': [1, 2, 3, 4, 5],
                'y': [2, 4, 6, 8, 10]
            })
            
            cas_table = self.cas_session.upload_frame(
                test_data,
                casout={'name': 'test_table', 'replace': True}
            )
            
            table_info = self.cas_session.table.tableInfo(name='test_table')
            
            print(f"Data Upload Test Passed")
            print(f"Test Table Size: {table_info['TableInfo']['Rows'][0]} Rows")
            
            self.cas_session.table.dropTable(name='test_table')
            print("Cleanup")
            
            return True
            
        except Exception as e:
            print(f"CAS Test Error: {e}")
            return False

    def close_connections(self):
        
        if self.cas_session:
            self.cas_session.close()
            print("CAS Connection Closed")
        
        self.is_connected = False
        print("All Connections Closed")