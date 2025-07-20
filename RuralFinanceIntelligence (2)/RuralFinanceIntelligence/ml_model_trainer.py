import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
# import lightgbm as lgb  # Temporarily disabled due to system dependency issue
import joblib
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class AdvancedPortfolioMLTrainer:
    """
    Advanced ML model trainer for rural Indian financial advisory system
    Uses ensemble methods to predict optimal portfolio allocations
    """
    
    def __init__(self):
        self.setup_logging()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare and engineer features for model training
        """
        self.logger.info("Preparing features for model training...")
        
        # Create feature engineering pipeline
        feature_df = df.copy()
        
        # 1. Create age-based features
        feature_df['Age_Group'] = pd.cut(feature_df['Age'], 
                                       bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['Young', 'Early_Career', 'Mid_Career', 'Pre_Retirement', 'Senior'])
        
        # 2. Income-based features
        feature_df['Total_Annual_Income'] = (feature_df['Monthly_Income_Primary'] + 
                                           feature_df['Monthly_Income_Secondary']) * 12
        feature_df['Income_Stability_Score'] = feature_df['Income_Stability'].map({
            'Fixed': 5, 'Stable': 4, 'Seasonal-Stable': 3, 'Variable': 2, 'Seasonal': 1
        })
        
        # 3. Risk capacity score
        feature_df['Risk_Capacity_Score'] = self._calculate_risk_capacity_score(feature_df)
        
        # 4. Investment readiness score
        feature_df['Investment_Readiness_Score'] = self._calculate_investment_readiness_score(feature_df)
        
        # 5. Accessibility score
        feature_df['Accessibility_Score'] = self._calculate_accessibility_score(feature_df)
        
        # 6. Government scheme eligibility count
        scheme_cols = [col for col in feature_df.columns if '_Eligible' in col]
        feature_df['Govt_Scheme_Eligibility_Count'] = feature_df[scheme_cols].sum(axis=1)
        
        # Define feature categories
        numerical_features = [
            'Age', 'Dependents', 'Monthly_Income_Primary', 'Monthly_Income_Secondary',
            'Total_Annual_Income', 'Land_Owned_Acres', 'Livestock_Value', 'Gold_Grams',
            'Total_Debt_Amount', 'Debt_Interest_Rate', 'Financial_Literacy_Score',
            'Short_Term_Horizon_Months', 'Long_Term_Horizon_Years', 'Bank_Distance_KM',
            'Electricity_Hours_Daily', 'Debt_to_Income_Ratio', 'Total_Asset_Value',
            'Net_Worth', 'Financial_Stability_Score', 'Monthly_Investment_Capacity',
            'Income_Stability_Score', 'Risk_Capacity_Score', 'Investment_Readiness_Score',
            'Accessibility_Score', 'Govt_Scheme_Eligibility_Count', 'Govt_Scheme_Awareness'
        ]
        
        categorical_features = [
            'Persona', 'Gender', 'Marital_Status', 'Education_Level', 'State',
            'Occupation', 'Community', 'Bank_Account_Type', 'Debt_Source',
            'Vehicle_Owned', 'Short_Term_Goal', 'Long_Term_Goal', 'Risk_Tolerance',
            'Investment_Preference', 'Primary_Crop', 'Water_Access', 'Road_Connectivity',
            'Age_Group', 'Income_Stability', 'Bank_Access_Level', 'Digital_Literacy'
        ]
        
        boolean_features = [
            'UPI_Usage', 'Smartphone_Access', 'Internet_Access', 'BPL_Status',
            'Aadhaar_Linked_Bank'
        ] + scheme_cols
        
        # Select features that exist in the dataframe
        numerical_features = [f for f in numerical_features if f in feature_df.columns]
        categorical_features = [f for f in categorical_features if f in feature_df.columns]
        boolean_features = [f for f in boolean_features if f in feature_df.columns]
        
        # Convert boolean to int
        for col in boolean_features:
            feature_df[col] = feature_df[col].astype(int)
        
        self.logger.info(f"Prepared {len(numerical_features)} numerical, {len(categorical_features)} categorical, {len(boolean_features)} boolean features")
        
        return feature_df, numerical_features, categorical_features, boolean_features
    
    def _calculate_risk_capacity_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate risk capacity score based on financial stability"""
        score = np.zeros(len(df))
        
        # Income factor (30%)
        income_score = np.clip(df['Total_Annual_Income'] / 100000, 0, 5)
        score += income_score * 0.3
        
        # Debt factor (25%)
        debt_score = np.clip(5 - df['Debt_to_Income_Ratio'] * 2, 0, 5)
        score += debt_score * 0.25
        
        # Asset factor (20%)
        asset_score = np.clip(np.log1p(df['Total_Asset_Value']) / 2, 0, 5)
        score += asset_score * 0.2
        
        # Age factor (15%)
        age_score = np.where(df['Age'] < 35, 5, 
                           np.where(df['Age'] < 50, 4, 
                                  np.where(df['Age'] < 60, 3, 2)))
        score += age_score * 0.15
        
        # Stability factor (10%)
        stability_score = df['Income_Stability_Score']
        score += stability_score * 0.1
        
        return np.clip(score, 0, 10)
    
    def _calculate_investment_readiness_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate investment readiness score"""
        score = np.zeros(len(df))
        
        # Financial literacy (40%)
        score += df['Financial_Literacy_Score'] * 0.4
        
        # Investment capacity (30%)
        capacity_score = np.clip(df['Monthly_Investment_Capacity'] / 5000, 0, 10)
        score += capacity_score * 0.3
        
        # Digital literacy (20%)
        digital_score = df['Digital_Literacy'].map({
            'High': 10, 'Medium-High': 8, 'Medium': 6, 'Low-Medium': 4, 'Low': 2
        }).fillna(2)
        score += digital_score * 0.2
        
        # Banking access (10%)
        banking_score = df['Bank_Access_Level'].map({
            'Excellent': 10, 'Good': 8, 'Medium': 6, 'Basic': 4, 'Limited': 2
        }).fillna(4)
        score += banking_score * 0.1
        
        return np.clip(score, 0, 10)
    
    def _calculate_accessibility_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate accessibility score for financial services"""
        score = np.zeros(len(df))
        
        # Bank distance (30%)
        distance_score = np.clip(10 - df['Bank_Distance_KM'] / 5, 0, 10)
        score += distance_score * 0.3
        
        # Digital access (25%)
        digital_score = (df['Smartphone_Access'].astype(int) * 5 + 
                        df['Internet_Access'].astype(int) * 5)
        score += digital_score * 0.25
        
        # UPI usage (20%)
        upi_score = df['UPI_Usage'].astype(int) * 10
        score += upi_score * 0.2
        
        # Road connectivity (15%)
        road_score = df['Road_Connectivity'].map({
            'Excellent': 10, 'Good': 8, 'Average': 6, 'Poor': 3
        }).fillna(5)
        score += road_score * 0.15
        
        # Electricity (10%)
        electricity_score = np.clip(df['Electricity_Hours_Daily'] / 2.4, 0, 10)
        score += electricity_score * 0.1
        
        return np.clip(score, 0, 10)
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for portfolio allocation
        Based on professional risk-profiling matrix
        """
        self.logger.info("Creating target portfolio allocations...")
        
        # Enhanced portfolio allocation matrix
        PORTFOLIO_MATRIX = {
            ('Low', 'Low'): {
                'Equity_Large_Cap': 0.0, 'Equity_Mid_Cap': 0.0, 'ELSS': 0.0,
                'Debt_Govt_Scheme': 0.50, 'Debt_FD': 0.30, 'PPF': 0.10,
                'Gold': 0.10, 'Cash': 0.0
            },
            ('Low', 'Medium'): {
                'Equity_Large_Cap': 0.05, 'Equity_Mid_Cap': 0.0, 'ELSS': 0.05,
                'Debt_Govt_Scheme': 0.40, 'Debt_FD': 0.30, 'PPF': 0.10,
                'Gold': 0.10, 'Cash': 0.0
            },
            ('Medium', 'Low'): {
                'Equity_Large_Cap': 0.10, 'Equity_Mid_Cap': 0.05, 'ELSS': 0.05,
                'Debt_Govt_Scheme': 0.35, 'Debt_FD': 0.25, 'PPF': 0.10,
                'Gold': 0.10, 'Cash': 0.0
            },
            ('Medium', 'Medium'): {
                'Equity_Large_Cap': 0.25, 'Equity_Mid_Cap': 0.10, 'ELSS': 0.10,
                'Debt_Govt_Scheme': 0.20, 'Debt_FD': 0.20, 'PPF': 0.10,
                'Gold': 0.05, 'Cash': 0.0
            },
            ('Medium', 'High'): {
                'Equity_Large_Cap': 0.35, 'Equity_Mid_Cap': 0.15, 'ELSS': 0.15,
                'Debt_Govt_Scheme': 0.10, 'Debt_FD': 0.15, 'PPF': 0.05,
                'Gold': 0.05, 'Cash': 0.0
            },
            ('High', 'Low'): {
                'Equity_Large_Cap': 0.20, 'Equity_Mid_Cap': 0.10, 'ELSS': 0.10,
                'Debt_Govt_Scheme': 0.25, 'Debt_FD': 0.20, 'PPF': 0.10,
                'Gold': 0.05, 'Cash': 0.0
            },
            ('High', 'Medium'): {
                'Equity_Large_Cap': 0.40, 'Equity_Mid_Cap': 0.15, 'ELSS': 0.15,
                'Debt_Govt_Scheme': 0.10, 'Debt_FD': 0.10, 'PPF': 0.05,
                'Gold': 0.05, 'Cash': 0.0
            },
            ('High', 'High'): {
                'Equity_Large_Cap': 0.50, 'Equity_Mid_Cap': 0.25, 'ELSS': 0.15,
                'Debt_Govt_Scheme': 0.05, 'Debt_FD': 0.0, 'PPF': 0.0,
                'Gold': 0.05, 'Cash': 0.0
            }
        }
        
        def assign_portfolio(row):
            # Determine risk capacity category
            if row['Risk_Capacity_Score'] <= 3:
                risk_capacity = 'Low'
            elif row['Risk_Capacity_Score'] <= 7:
                risk_capacity = 'Medium'
            else:
                risk_capacity = 'High'
            
            # Risk tolerance mapping
            risk_tolerance_map = {'Low': 'Low', 'Medium': 'Medium', 'High': 'High'}
            risk_tolerance = risk_tolerance_map.get(row['Risk_Tolerance'], 'Low')
            
            # Get portfolio allocation
            portfolio_key = (risk_capacity, risk_tolerance)
            if portfolio_key not in PORTFOLIO_MATRIX:
                portfolio_key = ('Medium', 'Medium')  # Default
            
            allocations = PORTFOLIO_MATRIX[portfolio_key].copy()
            
            # Adjust for accessibility constraints
            if row['Accessibility_Score'] < 5:
                # Reduce equity, increase traditional instruments
                equity_total = allocations['Equity_Large_Cap'] + allocations['Equity_Mid_Cap']
                allocations['Equity_Large_Cap'] *= 0.5
                allocations['Equity_Mid_Cap'] *= 0.5
                freed_allocation = equity_total * 0.5
                allocations['Debt_FD'] += freed_allocation * 0.7
                allocations['Gold'] += freed_allocation * 0.3
            
            # Adjust for government scheme eligibility
            if row.get('PM_Kisan_Eligible', False):
                allocations['Debt_Govt_Scheme'] = min(1.0, allocations['Debt_Govt_Scheme'] + 0.1)
            
            # Normalize to ensure sum = 1
            total = sum(allocations.values())
            if total > 0:
                allocations = {k: v/total for k, v in allocations.items()}
            
            return pd.Series(allocations)
        
        # Apply portfolio assignment
        portfolio_df = df.apply(assign_portfolio, axis=1)
        
        # Add target columns to original dataframe
        result_df = pd.concat([df, portfolio_df], axis=1)
        
        self.logger.info(f"Created portfolio allocations for {len(result_df)} profiles")
        return result_df
    
    def train_ensemble_models(self, X_train, X_test, y_train, y_test, target_cols):
        """
        Train ensemble models for each asset class
        """
        self.logger.info("Training ensemble models...")
        
        models_config = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            # 'LightGBM': lgb.LGBMRegressor(
            #     n_estimators=200,
            #     max_depth=6,
            #     learning_rate=0.1,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=42,
            #     verbose=-1
            # )  # Temporarily disabled due to system dependency issue
        }
        
        for target in target_cols:
            self.logger.info(f"Training models for {target}...")
            
            y_train_target = y_train[target]
            y_test_target = y_test[target]
            
            target_models = {}
            target_performance = {}
            
            for model_name, model in models_config.items():
                try:
                    # Train model
                    model.fit(X_train, y_train_target)
                    
                    # Predictions
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)
                    
                    # Performance metrics
                    train_r2 = r2_score(y_train_target, train_pred)
                    test_r2 = r2_score(y_test_target, test_pred)
                    test_mae = mean_absolute_error(y_test_target, test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test_target, test_pred))
                    
                    target_models[model_name] = model
                    target_performance[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse
                    }
                    
                    self.logger.info(f"{model_name} - {target}: R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Error training {model_name} for {target}: {str(e)}")
            
            self.models[target] = target_models
            self.model_performance[target] = target_performance
            
            # Store feature importance for best model
            best_model_name = max(target_performance.keys(), 
                                key=lambda k: target_performance[k]['test_r2'])
            best_model = target_models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance[target] = {
                    'model': best_model_name,
                    'importance': best_model.feature_importances_.tolist()
                }
    
    def train_complete_system(self, dataset_path: str):
        """
        Train the complete ML system
        """
        self.logger.info("Starting complete ML system training...")
        
        # Load dataset
        self.logger.info(f"Loading dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        self.logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Prepare features
        feature_df, numerical_features, categorical_features, boolean_features = self.prepare_features(df)
        
        # Create target variables (portfolio allocations)
        target_df = self.create_target_variables(feature_df)
        
        # Define target columns (asset allocation percentages)
        target_cols = [
            'Equity_Large_Cap', 'Equity_Mid_Cap', 'ELSS',
            'Debt_Govt_Scheme', 'Debt_FD', 'PPF', 'Gold', 'Cash'
        ]
        
        # Prepare feature matrix
        all_features = numerical_features + categorical_features + boolean_features
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
                ('bool', 'passthrough', boolean_features)
            ]
        )
        
        # Fit preprocessor and transform features
        X = target_df[all_features]
        y = target_df[target_cols]
        
        X_transformed = preprocessor.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=0.2, random_state=42, stratify=target_df['Risk_Tolerance']
        )
        
        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train ensemble models
        self.train_ensemble_models(X_train, X_test, y_train, y_test, target_cols)
        
        # Save preprocessor and feature names
        self.preprocessor = preprocessor
        self.feature_names = all_features
        self.target_names = target_cols
        
        # Save everything
        self.save_complete_system()
        
        self.logger.info("✅ Complete ML system training finished!")
        
        # Print summary
        self.print_training_summary()
    
    def save_complete_system(self):
        """Save the complete trained system"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        model_path = f'trained_portfolio_models_{timestamp}.joblib'
        joblib.dump({
            'models': self.models,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'training_timestamp': timestamp
        }, model_path)
        
        # Save performance report
        report_path = f'model_performance_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump({
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'training_summary': {
                    'timestamp': timestamp,
                    'feature_count': len(self.feature_names),
                    'target_count': len(self.target_names)
                }
            }, f, indent=2)
        
        self.logger.info(f"Saved complete system to {model_path}")
        self.logger.info(f"Saved performance report to {report_path}")
        
        return model_path, report_path
    
    def print_training_summary(self):
        """Print training summary"""
        print("\n" + "="*80)
        print("🎯 MODEL TRAINING SUMMARY")
        print("="*80)
        
        for target in self.target_names:
            if target in self.model_performance:
                print(f"\n📊 {target.upper()} ALLOCATION MODELS:")
                print("-" * 50)
                
                performances = self.model_performance[target]
                for model_name, metrics in performances.items():
                    print(f"  {model_name:15} | R² = {metrics['test_r2']:.4f} | MAE = {metrics['test_mae']:.4f}")
                
                # Best model
                best_model = max(performances.keys(), key=lambda k: performances[k]['test_r2'])
                best_r2 = performances[best_model]['test_r2']
                print(f"  🏆 Best: {best_model} (R² = {best_r2:.4f})")
        
        print("\n" + "="*80)

if __name__ == '__main__':
    trainer = AdvancedPortfolioMLTrainer()
    
    # Check if dataset exists
    dataset_path = 'enhanced_rural_financial_dataset.csv'
    
    try:
        trainer.train_complete_system(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please run enhanced_data_generator.py first to generate the dataset.")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
