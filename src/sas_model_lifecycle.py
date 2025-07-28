import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from sasctl import pzmm
from sasctl.services import model_repository, model_management, files

try:
    from .config import Config
except ImportError:
    import config
    Config = config.Config

class SASModelLifecycle:
    def __init__(self, sas_connection):
        self.config = Config()
        self.sas_connection = sas_connection
        self.registered_models = {}
        self.best_model_name = None
        
    def prepare_model_for_sas_registration(self, model_name, X_train, feature_names):
        """Modeli SAS registration için hazırla"""
        print(f"\n{model_name} PREPARATION FOR SAS REGISTRATION")
        print("=" * 60)
        
        if not self.sas_connection.is_connected:
            print("No SAS Connection.")
            return False
        
        try:
            model_path = os.path.join(self.config.MODELS_PATH, f"{model_name}.joblib")
            trained_model = joblib.load(model_path)
            print(f"Model Uploaded: {model_path}")
            
            pzmm_path = os.path.join(self.config.MODELS_PATH, "pzmm_packages")
            os.makedirs(pzmm_path, exist_ok=True)
            
            model_prefix = f"{self.config.MODEL_MANAGER_CONFIG['model_prefix']}_{model_name}"
            
            print("Model packaging with PZMM")
            
            pzmm.PickleModel.pickle_trained_model(
                model_prefix=model_prefix,
                trained_model=trained_model,
                pickle_path=pzmm_path
            )
            
            print(f"PZMM package created: {pzmm_path}")
            
            model_metadata = {
                'name': model_prefix,
                'description': f'Bank customer churn prediction using {model_name}',
                'algorithm': type(trained_model).__name__,
                'target_variable': 'Churn',
                'input_variables': feature_names,
                'creation_date': datetime.now().isoformat(),
                'model_type': 'Classification'
            }
            
            self.registered_models[model_name] = {
                'model_prefix': model_prefix,
                'pzmm_path': pzmm_path,
                'metadata': model_metadata,
                'trained_model': trained_model
            }
            
            print(f"Model is Ready for Registration: {model_prefix}")
            return True
            
        except Exception as e:
            print(f"Model preparation error: {e}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            return False    

    def _delete_existing_file(self, filename, project_name):
        try:
            existing_files = files.list_files(filter=f"eq(name,'{filename}')")
            
            if existing_files:
                for existing_file in existing_files:
                    print(f"Existing file found: {filename}, deleting...")
                    files.delete_file(existing_file['id'])
                    print(f"Removed File: {filename}")
                    return True
                
        except Exception as e:
            print(f"{e}")
            
        return False

    def _upload_single_file(self, file_path, project_name, overwrite=True):
        """Simplified file upload with basic overwrite functionality"""
        filename = os.path.basename(file_path)
        
        try:
            # Delete existing file if overwrite is enabled
            if overwrite:
                self._delete_existing_file(filename, project_name)
            
            # Upload the file
            with open(file_path, 'rb') as f:
                uploaded_file = files.create_file(
                    file=f,
                    filename=filename,
                    folder=project_name
                )
            
            print(f"✓ Successful Upload: {filename}")
            return uploaded_file
            
        except Exception as upload_error:
            error_str = str(upload_error)
            
            # Handle file exists error with one retry
            if ("409" in error_str or "already exists" in error_str) and overwrite:
                print(f"File exists, retrying overwrite: {filename}")
                
                try:
                    # Force delete and retry once
                    self._delete_existing_file(filename, project_name)
                    
                    with open(file_path, 'rb') as f:
                        uploaded_file = files.create_file(
                            file=f,
                            filename=filename,
                            folder=project_name
                        )
                    
                    print(f"✓ Overwrite successful: {filename}")
                    return uploaded_file
                    
                except Exception as retry_error:
                    print(f"❌ Upload failed after retry: {retry_error}")
                    return None
            else:
                print(f"❌ Upload failed: {upload_error}")
                return None
        
    def _upload_pzmm_files(self, model_info, project_name, overwrite=True):
        """Upload PZMM files to SAS Model Manager"""
        pzmm_path = model_info['pzmm_path']
        model_prefix = model_info['model_prefix']
        
        import glob
        pzmm_files = glob.glob(os.path.join(pzmm_path, f"{model_prefix}*"))
        
        print(f"PZMM files: {len(pzmm_files)} files found")
        for file_path in pzmm_files:
            print(f"   - {os.path.basename(file_path)}")
        
        uploaded_files = []
        
        for file_path in pzmm_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                print(f"Uploading: {filename}")
                
                uploaded_file = self._upload_single_file(file_path, project_name, overwrite)
                
                if uploaded_file:
                    uploaded_files.append(uploaded_file)
                else:
                    print(f"{filename} upload failed, continuing...")
        
        return uploaded_files
    
    def _create_model_variables(self, model_info):
        """Create input/output variables for SAS model"""
        feature_names = model_info['metadata']['input_variables']
        
        input_variables = []
        for feature in feature_names:
            input_variables.append({
                'name': feature,
                'description': f'Input feature: {feature}',
                'role': 'input',
                'type': 'decimal',
                'level': 'interval',
                'format': '',
                'aggregation': '',
                'length': 8
            })
        
        output_variables = [
            {
                'name': 'P_Churn1',
                'description': 'Predicted probability of Churn=1',
                'role': 'output',
                'type': 'decimal',
                'level': 'interval',
                'format': '',
                'aggregation': '',
                'length': 8
            },
            {
                'name': 'P_Churn0', 
                'description': 'Predicted probability of Churn=0',
                'role': 'output',
                'type': 'decimal',
                'level': 'interval',
                'format': '',
                'aggregation': '',
                'length': 8
            },
            {
                'name': 'I_Churn',
                'description': 'Predicted class for Churn',
                'role': 'output',
                'type': 'string',
                'level': 'nominal',
                'format': '',
                'aggregation': '',
                'length': 32
            },
            {
                'name': '_WARN_',
                'description': 'Warning messages',
                'role': 'output',
                'type': 'string', 
                'level': 'nominal',
                'format': '',
                'aggregation': '',
                'length': 200
            }
        ]
        
        return input_variables, output_variables
    
    def _create_model_in_sas(self, model_info, project_name, input_variables, output_variables, is_champion=False):
        """Create model in SAS Model Manager"""
        model_prefix = model_info['model_prefix']
        
        print(f"🔗 Model registering (Champion: {is_champion})...")
        
        model_obj = model_repository.create_model(
            model=model_prefix,
            project=project_name,
            description=model_info['metadata']['description'],
            modeler='Python',
            algorithm=model_info['metadata']['algorithm'],
            function='classification',
            target_variable='Churn',
            tool='Python 3',
            is_champion=is_champion,
            score_code_type='DS2',
            input_variables=input_variables,
            output_variables=output_variables
        )
        
        model_id = model_obj.get('id') if isinstance(model_obj, dict) else getattr(model_obj, 'id', None)
        print(f"✅ Model created, ID: {model_id}")
        
        return model_id
    
    def _set_champion_status(self, model_id, model_name, project_name, is_champion):
        """Set champion status for model"""
        if not is_champion or not model_id:
            return True
            
        try:
            print(f"🏆 Setting {model_name} as CHAMPION model...")
            
            # Method 1: Get the current model object and update it
            current_model = model_repository.get_model(model_id)
            
            if current_model:
                # Update the model object with champion status
                current_model['champion'] = True
                current_model['isChampion'] = True
                
                # Update the model with the modified object
                model_repository.update_model(current_model)
                
                print(f"🏆 ✅ {model_name} successfully set as CHAMPION!")
                return True
            else:
                print(f"❌ Could not retrieve model object for champion update")
                return False
                
        except Exception as champion_error:
            print(f"⚠️  Champion setting failed: {champion_error}")
            try:
                # Method 2: Try setting champion at project level
                project_obj = model_repository.get_project(project_name)
                if project_obj:
                    project_obj['championModelId'] = model_id
                    project_obj['championModel'] = model_id
                    model_repository.update_project(project_obj)
                    print(f"🏆 ✅ Champion status set via project update!")
                    return True
                else:
                    print(f"❌ Could not retrieve project object")
                    return False
            except Exception as alt_error:
                print(f"❌ Project-level champion method also failed: {alt_error}")
                return False
    
    def _attach_files_to_model(self, model_id, uploaded_files):
        """Attach uploaded files to model"""
        if not model_id or not uploaded_files:
            return 0
            
        print(f"🔗 {len(uploaded_files)} files attaching to model...")
        
        attached_count = 0
        for uploaded_file in uploaded_files:
            try:
                file_id = uploaded_file.get('id') if isinstance(uploaded_file, dict) else getattr(uploaded_file, 'id', None)
                file_name = uploaded_file.get('name') if isinstance(uploaded_file, dict) else getattr(uploaded_file, 'name', 'Unknown')
                
                if file_id:
                    print(f"File attaching: {file_name}")
                    
                    model_repository.add_model_content(
                        model=model_id,
                        file=file_id,
                        name=file_name,
                        role='score'
                    )
                    
                    attached_count += 1
                    print(f"✅ File attached: {file_name}")
                else:
                    print(f"⚠️  File ID not found: {file_name}")
                    
            except Exception as attach_error:
                print(f"❌ File attaching error ({file_name}): {attach_error}")
        
        return attached_count
    
    def register_model_to_sas_manager(self, model_name, overwrite=True):
        """Streamlined model registration using helper methods"""
        print(f"\n{model_name} Registering to SAS Model Manager...")
        print("=" * 60)
        
        if model_name not in self.registered_models:
            return False
        
        model_info = self.registered_models[model_name]
        project_name = self.config.MODEL_MANAGER_CONFIG['project_name']
        
        is_champion = (model_name == self.best_model_name)
        champion_status = "🏆 CHAMPION" if is_champion else "📊 STANDARD"
        print(f"Model Status: {champion_status}")
        print(f"Overwrite mode: {'Open' if overwrite else 'Closed'}")
        
        try:
            # 1. Upload PZMM files
            uploaded_files = self._upload_pzmm_files(model_info, project_name, overwrite)
            
            if not uploaded_files:
                print("❌ No files uploaded")
                return False
            
            print(f"\n✅ {len(uploaded_files)} files uploaded!")
            
            # 2. Create model variables
            input_variables, output_variables = self._create_model_variables(model_info)
            
            print(f"📊 Input Variables: {len(input_variables)} features")
            print(f"📊 Output Variables: {len(output_variables)} outputs")
            
            # 3. Create model in SAS
            model_id = self._create_model_in_sas(model_info, project_name, input_variables, output_variables, is_champion)
            
            if not model_id:
                print("❌ Model creation failed")
                return False
            
            # 4. Set champion status
            champion_success = self._set_champion_status(model_id, model_name, project_name, is_champion)
            
            # 5. Attach files to model
            attached_count = self._attach_files_to_model(model_id, uploaded_files)
            
            # Final summary
            print(f"\n✅ Model registration completed!")
            print(f"Model: {model_info['model_prefix']}")
            print(f"Project: {project_name}")
            print(f"Model ID: {model_id}")
            print(f"Champion Status: {'✅ CHAMPION' if is_champion and champion_success else '📊 STANDARD'}")
            print(f"Input Variables: {len(input_variables)} features")
            print(f"Output Variables: {len(output_variables)} outputs")
            print(f"Attached files: {attached_count}")
            
            return True
            
        except Exception as reg_error:
            reg_error_str = str(reg_error)
            print(f"❌ Model registration error: {reg_error}")
            
            if "already exists" in reg_error_str or "409" in reg_error_str:
                print(f"⚠️  Model already registered: {model_info['model_prefix']}")
                print("✅ Files updated, existing model will be used")
                return True
            else:
                print(f"⚠️  Model registration failed: {reg_error}")
                return False
    
    def score_new_data_with_cas(self, new_data, model_name):
        print(f"\nNEW DATA SCORING WITH CAS ({model_name})...")
        print("=" * 60)
        
        if not self.sas_connection.cas_session:
            print("CAS connection not found!")
            return None
        
        if model_name not in self.registered_models:
            print(f"Model {model_name} not registered!")
            return None
        
        try:
            print("Data uploading to CAS...")
            
            cas_table = self.sas_connection.cas_session.upload_frame(
                new_data,
                casout={'name': 'scoring_data', 'replace': True}
            )
            
            print(f"{len(new_data)} rows uploaded to CAS")
            
            model_info = self.registered_models[model_name]
            trained_model = model_info['trained_model']
            
            print("Scoring process...")
            
            predictions = trained_model.predict_proba(new_data)[:, 1]  # Churn probability
            
            scoring_results = {
                'table_name': 'scoring_data',
                'model_used': model_info['model_prefix'],
                'records_scored': len(new_data),
                'timestamp': datetime.now().isoformat(),
                'mean_churn_probability': float(predictions.mean()),
                'high_risk_customers': int((predictions > 0.5).sum())
            }
            
            print(f"Scoring completed!")
            print(f"Records scored: {scoring_results['records_scored']}")
            print(f"Mean churn probability: {scoring_results['mean_churn_probability']:.3f}")
            print(f"High risk customers: {scoring_results['high_risk_customers']}")
            
            return scoring_results
            
        except Exception as e:
            print(f"CAS scoring error: {e}")
            return None
    
    def generate_sas_reports(self, model_name):
        print(f"\nSAS REPORTS GENERATION ({model_name})...")
        print("=" * 60)
        
        if model_name not in self.registered_models:
            print(f"Model {model_name} not registered!")
            return False
        
        try:
            model_info = self.registered_models[model_name]
            
            is_champion = (model_name == self.best_model_name)
            
            performance_summary = {
                'model_name': model_info['model_prefix'],
                'registration_time': datetime.now().isoformat(),
                'status': 'Champion' if is_champion else 'Standard',
                'algorithm': model_info['metadata']['algorithm'],
                'target_variable': model_info['metadata']['target_variable'],
                'features_count': len(model_info['metadata']['input_variables']),
                'model_type': model_info['metadata']['model_type'],
                'is_champion': is_champion
            }
            
            reports_path = os.path.join(self.config.REPORTS_PATH, "sas_reports")
            os.makedirs(reports_path, exist_ok=True)
            
            report_file = os.path.join(reports_path, f"{model_name}_sas_report.json")
            with open(report_file, 'w') as f:
                json.dump(performance_summary, f, indent=2)
            
            champion_text = " (🏆 CHAMPION)" if is_champion else ""
            print(f"SAS report created: {report_file}")
            print(f"Model: {performance_summary['model_name']}{champion_text}")
            print(f"Algorithm: {performance_summary['algorithm']}")
            print(f"Features: {performance_summary['features_count']}")
            
            return True
            
        except Exception as e:
            print(f"SAS report error: {e}")
            return False
    
    def register_all_models_to_sas(self, model_list, X_train, feature_names, best_model_name, overwrite=True):
        print("\n🔄 ALL MODELS REGISTRATION TO SAS...")
        print("=" * 70)
        
        self.best_model_name = best_model_name
        print(f"🏆 Champion Model: {best_model_name}")
        
        successful_models = []
        failed_models = []
        
        for i, model_name in enumerate(model_list, 1):
            champion_indicator = " 🏆" if model_name == best_model_name else ""
            print(f"\n📦 MODEL {i}/{len(model_list)}: {model_name}{champion_indicator}")
            print("-" * 50)
            
            try:
                if self.prepare_model_for_sas_registration(model_name, X_train, feature_names):
                    if self.register_model_to_sas_manager(model_name, overwrite=overwrite):
                        successful_models.append(model_name)
                        success_text = "✅ CHAMPION registered" if model_name == best_model_name else "✅ successfully registered"
                        print(f"{model_name} {success_text}")
                    else:
                        failed_models.append(model_name)
                        print(f"❌ {model_name} not registered")
                else:
                    failed_models.append(model_name)
                    print(f"❌ {model_name} not prepared")
                    
            except Exception as e:
                failed_models.append(model_name)
                print(f"❌ {model_name} error: {e}")
        
        print(f"\n📊 ALL MODELS SUMMARY:")
        print("=" * 50)
        print(f"✅ Successful: {len(successful_models)} model")
        for model in successful_models:
            champion_text = " 🏆 (CHAMPION)" if model == best_model_name else ""
            print(f"   - {model}{champion_text}")
        
        if failed_models:
            print(f"❌ Failed: {len(failed_models)} model")
            for model in failed_models:
                print(f"   - {model}")
        
        return successful_models, failed_models

    def generate_all_reports(self, model_list):
        print(f"\n📊 ALL MODELS REPORTS...")
        print("=" * 70)
        
        successful_reports = []
        failed_reports = []
        
        for model_name in model_list:
            try:
                if model_name in self.registered_models:
                    if self.generate_sas_reports(model_name):
                        successful_reports.append(model_name)
                        champion_text = " 🏆" if model_name == self.best_model_name else ""
                        print(f"✅ {model_name}{champion_text} report created")
                    else:
                        failed_reports.append(model_name)
                        print(f"❌ {model_name} report not created")
                else:
                    failed_reports.append(model_name)
                    print(f"⚠️  {model_name} not registered, report skipped")
            except Exception as e:
                failed_reports.append(model_name)
                print(f"❌ {model_name} report error: {e}")
        
        print(f"\n📊 REPORT SUMMARY: ✅{len(successful_reports)} ❌{len(failed_reports)}")
        return successful_reports, failed_reports

    def full_sas_lifecycle_pipeline(self, best_model_name, X_train, feature_names, sample_new_data=None, overwrite=True, upload_all_models=True):
        """Streamlined SAS lifecycle pipeline"""
        print("\n🚀 SAS MODEL LIFECYCLE PIPELINE")
        print("=" * 70)
        
        success_steps = []
        self.best_model_name = best_model_name
        
        # Step 1: Model Registration
        if upload_all_models:
            print("\n1️⃣ REGISTERING ALL MODELS...")
            available_models = ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting']
            
            successful_models, failed_models = self.register_all_models_to_sas(
                available_models, X_train, feature_names, best_model_name, overwrite
            )
            
            if successful_models:
                success_steps.append("Models Registration")
                if best_model_name in successful_models:
                    success_steps.append("Champion Selection")
        else:
            print("\n1️⃣ REGISTERING SINGLE MODEL...")
            if (self.prepare_model_for_sas_registration(best_model_name, X_train, feature_names) and
                self.register_model_to_sas_manager(best_model_name, overwrite=overwrite)):
                success_steps.append("Registration")
                successful_models = [best_model_name]
            else:
                successful_models = []
        
        # Step 2: Reports Generation
        if successful_models:
            print("\n2️⃣ GENERATING REPORTS...")
            successful_reports, failed_reports = self.generate_all_reports(successful_models)
            if successful_reports:
                success_steps.append("Reports")
        
        # Step 3: Scoring Test (Optional)
        if sample_new_data is not None and best_model_name in self.registered_models:
            print("\n3️⃣ SCORING TEST...")
            if self.score_new_data_with_cas(sample_new_data, best_model_name):
                success_steps.append("Scoring")
        
        # Final Summary
        print(f"\n{'='*70}")
        print("🎯 PIPELINE RESULTS")
        print(f"{'='*70}")
        print(f"✅ Successful steps: {', '.join(success_steps) if success_steps else 'None'}")
        print(f"📊 Models registered: {len(successful_models) if 'successful_models' in locals() else 0}")
        print(f"🏆 Champion model: {best_model_name}")
        
        success = len(success_steps) >= 1
        print(f"🎉 Pipeline {'COMPLETED' if success else 'FAILED'}!")
        return success