import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import json

class EnhancedRuralDataGenerator:
    """
    Enhanced data generator for rural India financial profiles
    Incorporates all 7 parameter categories with realistic correlations
    """
    
    def __init__(self):
        self.setup_logging()
        self.load_configurations()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_configurations(self):
        """Load all configuration data for realistic profile generation"""
        
        # 1. Enhanced Micro-Personas with detailed attributes
        self.personas_config = {
            'Marginal_Farmer_Bihar': {
                'prevalence': 0.15,
                'age_range': (25, 70),
                'income_range': (4000, 12000),
                'edu_dist': [0.6, 0.35, 0.05, 0.0],
                'states': ['Bihar', 'Jharkhand', 'West Bengal'],
                'occupations': ['Marginal Farmer', 'Agricultural Laborer'],
                'income_stability': 'Seasonal',
                'bank_access': 'Limited',
                'digital_literacy': 'Low'
            },
            'Irrigated_Farmer_Punjab': {
                'prevalence': 0.10,
                'age_range': (30, 65),
                'income_range': (20000, 60000),
                'edu_dist': [0.2, 0.5, 0.25, 0.05],
                'states': ['Punjab', 'Haryana', 'Uttar Pradesh'],
                'occupations': ['Irrigated Farmer', 'Dairy Farmer'],
                'income_stability': 'Seasonal-Stable',
                'bank_access': 'Good',
                'digital_literacy': 'Medium'
            },
            'Rural_Laborer_UP': {
                'prevalence': 0.20,
                'age_range': (18, 60),
                'income_range': (5000, 15000),
                'edu_dist': [0.7, 0.25, 0.05, 0.0],
                'states': ['Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan'],
                'occupations': ['Daily Wage Laborer', 'Construction Worker', 'MGNREGA Worker'],
                'income_stability': 'Variable',
                'bank_access': 'Basic',
                'digital_literacy': 'Low'
            },
            'Small_Town_Trader_MH': {
                'prevalence': 0.20,
                'age_range': (25, 60),
                'income_range': (25000, 100000),
                'edu_dist': [0.1, 0.3, 0.5, 0.1],
                'states': ['Maharashtra', 'Gujarat', 'Karnataka'],
                'occupations': ['Small Business Owner', 'Trader', 'Shop Owner'],
                'income_stability': 'Stable',
                'bank_access': 'Good',
                'digital_literacy': 'Medium-High'
            },
            'Urban_Gig_Worker_KA': {
                'prevalence': 0.15,
                'age_range': (20, 40),
                'income_range': (20000, 90000),
                'edu_dist': [0.0, 0.1, 0.6, 0.3],
                'states': ['Karnataka', 'Tamil Nadu', 'Telangana'],
                'occupations': ['Delivery Partner', 'Cab Driver', 'Freelancer'],
                'income_stability': 'Variable',
                'bank_access': 'Excellent',
                'digital_literacy': 'High'
            },
            'Salaried_Formal_TN': {
                'prevalence': 0.10,
                'age_range': (24, 58),
                'income_range': (40000, 300000),
                'edu_dist': [0.0, 0.05, 0.4, 0.55],
                'states': ['Tamil Nadu', 'Kerala', 'Andhra Pradesh'],
                'occupations': ['Government Employee', 'Private Sector Employee', 'Teacher'],
                'income_stability': 'Stable',
                'bank_access': 'Excellent',
                'digital_literacy': 'High'
            },
            'Retired_Pensioner': {
                'prevalence': 0.10,
                'age_range': (60, 85),
                'income_range': (8000, 50000),
                'edu_dist': [0.2, 0.4, 0.3, 0.1],
                'states': ['All States'],
                'occupations': ['Retired'],
                'income_stability': 'Fixed',
                'bank_access': 'Good',
                'digital_literacy': 'Low-Medium'
            }
        }
        
        # 2. State-wise economic data
        self.state_economic_data = {
            'Bihar': {'avg_income_multiplier': 0.7, 'bank_density': 'Low', 'internet_penetration': 0.3},
            'Punjab': {'avg_income_multiplier': 1.3, 'bank_density': 'High', 'internet_penetration': 0.8},
            'Uttar Pradesh': {'avg_income_multiplier': 0.8, 'bank_density': 'Medium', 'internet_penetration': 0.5},
            'Maharashtra': {'avg_income_multiplier': 1.4, 'bank_density': 'High', 'internet_penetration': 0.9},
            'Karnataka': {'avg_income_multiplier': 1.5, 'bank_density': 'High', 'internet_penetration': 0.9},
            'Tamil Nadu': {'avg_income_multiplier': 1.3, 'bank_density': 'High', 'internet_penetration': 0.8},
        }
        
        # 3. Government scheme eligibility matrix
        self.govt_schemes = {
            'PM_Kisan': {'income_threshold': 50000, 'land_requirement': True},
            'PMJJBY': {'age_range': (18, 50), 'bank_account': True},
            'PMSBY': {'age_range': (18, 70), 'bank_account': True},
            'APY': {'age_range': (18, 40), 'unorganized_sector': True},
            'PMFBY': {'farmer': True, 'crop_insurance': True},
            'Mudra_Loan': {'business_owner': True, 'loan_amount_range': (50000, 1000000)}
        }
    
    def generate_ultimate_dataset(self, num_rows: int = 10000000) -> pd.DataFrame:
        """
        Generate comprehensive dataset with all 7 parameter categories
        """
        self.logger.info(f"Generating {num_rows} enhanced rural financial profiles...")
        
        # Generate base profiles
        profile_list = []
        for persona, props in self.personas_config.items():
            count = int(num_rows * props['prevalence'])
            
            # Basic demographics
            ages = np.random.randint(props['age_range'][0], props['age_range'][1], size=count)
            
            profile = {
                'Persona': np.full(count, persona),
                'Age': ages,
                'Gender': np.random.choice(['Male', 'Female'], count, p=[0.65, 0.35]),
                'Marital_Status': self._generate_marital_status(ages),
                'Education_Level': np.random.choice(
                    ['Illiterate', 'Primary', 'Secondary', 'Graduate'], 
                    count, 
                    p=props['edu_dist']
                ),
                'State': np.random.choice(props['states'], count),
                'Occupation': np.random.choice(props['occupations'], count),
                'Income_Stability': np.full(count, props['income_stability']),
                'Bank_Access_Level': np.full(count, props['bank_access']),
                'Digital_Literacy': np.full(count, props['digital_literacy'])
            }
            
            # Income parameters with seasonality
            base_income = np.random.randint(props['income_range'][0], props['income_range'][1], size=count)
            profile['Monthly_Income_Primary'] = base_income
            profile['Monthly_Income_Secondary'] = self._generate_secondary_income(base_income, persona)
            profile['Income_Seasonality_Factor'] = self._generate_seasonality_factor(props['income_stability'], count)
            
            profile_list.append(pd.DataFrame(profile))
        
        # Combine all profiles
        df = pd.concat(profile_list, ignore_index=True).sample(frac=1).reset_index(drop=True)
        
        # Generate comprehensive features
        df = self._add_demographic_features(df)
        df = self._add_asset_liability_features(df)
        df = self._add_banking_financial_literacy_features(df)
        df = self._add_financial_goals_features(df)
        df = self._add_location_seasonal_features(df)
        df = self._add_government_scheme_eligibility(df)
        df = self._add_derived_financial_metrics(df)
        
        self.logger.info(f"Generated dataset with {len(df)} rows and {len(df.columns)} features")
        return df
    
    def _generate_marital_status(self, ages: np.ndarray) -> np.ndarray:
        """Generate realistic marital status based on age"""
        marital_status = []
        for age in ages:
            if age < 20:
                status = np.random.choice(['Single'], p=[1.0])
            elif age < 25:
                status = np.random.choice(['Single', 'Married'], p=[0.6, 0.4])
            elif age < 45:
                status = np.random.choice(['Single', 'Married'], p=[0.1, 0.9])
            else:
                status = np.random.choice(['Married', 'Widowed'], p=[0.85, 0.15])
            marital_status.append(status)
        return np.array(marital_status)
    
    def _generate_secondary_income(self, primary_income: np.ndarray, persona: str) -> np.ndarray:
        """Generate secondary income based on persona and primary income"""
        secondary_multipliers = {
            'Marginal_Farmer_Bihar': (0.1, 0.3),
            'Irrigated_Farmer_Punjab': (0.2, 0.5),
            'Rural_Laborer_UP': (0.0, 0.2),
            'Small_Town_Trader_MH': (0.3, 0.8),
            'Urban_Gig_Worker_KA': (0.1, 0.4),
            'Salaried_Formal_TN': (0.0, 0.1),
            'Retired_Pensioner': (0.0, 0.2)
        }
        
        min_mult, max_mult = secondary_multipliers.get(persona, (0.0, 0.1))
        multipliers = np.random.uniform(min_mult, max_mult, len(primary_income))
        return (primary_income * multipliers).astype(int)
    
    def _generate_seasonality_factor(self, income_stability: str, count: int) -> np.ndarray:
        """Generate income seasonality factors"""
        if income_stability == 'Seasonal':
            return np.random.uniform(0.5, 1.5, count)
        elif income_stability == 'Seasonal-Stable':
            return np.random.uniform(0.8, 1.2, count)
        elif income_stability == 'Variable':
            return np.random.uniform(0.6, 1.4, count)
        else:  # Stable or Fixed
            return np.random.uniform(0.95, 1.05, count)
    
    def _add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add detailed demographic features"""
        # Number of dependents based on age and marital status
        df['Dependents'] = df.apply(lambda row: self._calculate_dependents(row), axis=1)
        
        # Caste/Community (for scheme eligibility)
        df['Community'] = np.random.choice(['General', 'OBC', 'SC', 'ST'], len(df), p=[0.3, 0.4, 0.2, 0.1])
        
        return df
    
    def _calculate_dependents(self, row) -> int:
        """Calculate realistic number of dependents"""
        if row['Age'] < 25 or row['Marital_Status'] == 'Single':
            return np.random.randint(0, 2)
        elif row['Age'] < 35:
            return np.random.randint(1, 4)
        elif row['Age'] < 50:
            return np.random.randint(2, 6)
        else:
            return np.random.randint(1, 3)
    
    def _add_asset_liability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive asset and liability features"""
        # Land ownership
        df['Land_Owned_Acres'] = df['Persona'].apply(self._generate_land_ownership)
        
        # Livestock assets
        df['Livestock_Value'] = df.apply(lambda row: self._generate_livestock_value(row), axis=1)
        
        # Gold/Silver owned (common rural asset)
        df['Gold_Grams'] = df.apply(lambda row: self._generate_gold_ownership(row), axis=1)
        
        # Vehicle ownership
        df['Vehicle_Owned'] = df.apply(lambda row: self._generate_vehicle_ownership(row), axis=1)
        
        # Debt details
        total_income = df['Monthly_Income_Primary'] + df['Monthly_Income_Secondary']
        df['Total_Debt_Amount'] = (total_income * 12 * np.random.uniform(0.1, 3.0, len(df))).astype(int)
        df['Debt_Source'] = np.random.choice(['Bank', 'SHG', 'Moneylender', 'Family', 'Multiple'], len(df))
        df['Debt_Interest_Rate'] = df['Debt_Source'].apply(self._assign_interest_rate)
        
        return df
    
    def _generate_land_ownership(self, persona: str) -> float:
        """Generate realistic land ownership based on persona"""
        if 'Farmer' in persona:
            if 'Marginal' in persona:
                return np.random.uniform(0.5, 2.5)
            else:
                return np.random.uniform(2.0, 10.0)
        elif 'Rural' in persona:
            return np.random.uniform(0, 1.0) if np.random.random() < 0.3 else 0
        else:
            return 0
    
    def _generate_livestock_value(self, row) -> int:
        """Generate livestock value based on persona and location"""
        if 'Farmer' in row['Persona'] or 'Rural' in row['Persona']:
            base_value = np.random.randint(5000, 50000)
            return base_value if np.random.random() < 0.6 else 0
        return 0
    
    def _generate_gold_ownership(self, row) -> float:
        """Generate gold ownership (common rural asset class)"""
        # Higher gold ownership among married women and farmers
        base_grams = np.random.uniform(10, 100)
        if row['Gender'] == 'Female' and row['Marital_Status'] == 'Married':
            base_grams *= 2
        if row['Monthly_Income_Primary'] > 20000:
            base_grams *= 1.5
        return base_grams
    
    def _generate_vehicle_ownership(self, row) -> str:
        """Generate vehicle ownership based on income and persona"""
        income = row['Monthly_Income_Primary']
        if income < 10000:
            return np.random.choice(['None', 'Bicycle'], p=[0.6, 0.4])
        elif income < 25000:
            return np.random.choice(['None', 'Bicycle', 'Motorcycle'], p=[0.3, 0.3, 0.4])
        elif income < 50000:
            return np.random.choice(['Bicycle', 'Motorcycle', 'Car'], p=[0.2, 0.6, 0.2])
        else:
            return np.random.choice(['Motorcycle', 'Car', 'Tractor'], p=[0.4, 0.4, 0.2])
    
    def _assign_interest_rate(self, debt_source: str) -> float:
        """Assign realistic interest rates based on debt source"""
        rates = {
            'Bank': np.random.uniform(8, 12),
            'SHG': np.random.uniform(12, 18),
            'Moneylender': np.random.uniform(24, 60),
            'Family': np.random.uniform(0, 6),
            'Multiple': np.random.uniform(12, 24)
        }
        return rates.get(debt_source, 15)
    
    def _add_banking_financial_literacy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add banking and financial literacy features"""
        # Bank account types
        df['Bank_Account_Type'] = df.apply(lambda row: self._assign_bank_account_type(row), axis=1)
        
        # Digital payment usage
        df['UPI_Usage'] = df.apply(lambda row: self._assign_upi_usage(row), axis=1)
        
        # Smartphone access
        df['Smartphone_Access'] = df.apply(lambda row: self._assign_smartphone_access(row), axis=1)
        
        # Financial literacy score
        df['Financial_Literacy_Score'] = df.apply(lambda row: self._calculate_financial_literacy(row), axis=1)
        
        # Government scheme awareness
        df['Govt_Scheme_Awareness'] = df.apply(lambda row: self._assign_scheme_awareness(row), axis=1)
        
        # Bank branch proximity
        df['Bank_Distance_KM'] = df.apply(lambda row: self._assign_bank_distance(row), axis=1)
        
        return df
    
    def _assign_bank_account_type(self, row) -> str:
        """Assign bank account type based on income and demographics"""
        if row['Monthly_Income_Primary'] < 10000:
            return np.random.choice(['Jan Dhan', 'Basic Savings'], p=[0.7, 0.3])
        elif row['Monthly_Income_Primary'] < 30000:
            return np.random.choice(['Basic Savings', 'Regular Savings'], p=[0.6, 0.4])
        else:
            return np.random.choice(['Regular Savings', 'Current'], p=[0.8, 0.2])
    
    def _assign_upi_usage(self, row) -> bool:
        """Assign UPI usage based on digital literacy and demographics"""
        if row['Digital_Literacy'] == 'High':
            return np.random.choice([True, False], p=[0.9, 0.1])
        elif row['Digital_Literacy'] == 'Medium-High':
            return np.random.choice([True, False], p=[0.8, 0.2])
        elif row['Digital_Literacy'] == 'Medium':
            return np.random.choice([True, False], p=[0.6, 0.4])
        else:
            return np.random.choice([True, False], p=[0.2, 0.8])
    
    def _assign_smartphone_access(self, row) -> bool:
        """Assign smartphone access"""
        if row['Age'] < 30:
            return np.random.choice([True, False], p=[0.8, 0.2])
        elif row['Age'] < 50:
            return np.random.choice([True, False], p=[0.6, 0.4])
        else:
            return np.random.choice([True, False], p=[0.3, 0.7])
    
    def _calculate_financial_literacy(self, row) -> int:
        """Calculate financial literacy score (1-10)"""
        base_score = {
            'Illiterate': 1,
            'Primary': 3,
            'Secondary': 6,
            'Graduate': 8
        }.get(row['Education_Level'], 3)
        
        # Adjust based on occupation and age
        if 'Business' in row['Occupation'] or 'Trader' in row['Occupation']:
            base_score += 2
        if row['Age'] > 40:
            base_score += 1
        if row['Digital_Literacy'] == 'High':
            base_score += 1
            
        return min(10, max(1, base_score + np.random.randint(-1, 2)))
    
    def _assign_scheme_awareness(self, row) -> int:
        """Assign government scheme awareness score (0-10)"""
        base_awareness = {
            'Low': 2,
            'Medium': 5,
            'Medium-High': 7,
            'High': 8
        }.get(row['Digital_Literacy'], 2)
        
        return min(10, max(0, base_awareness + np.random.randint(-2, 3)))
    
    def _assign_bank_distance(self, row) -> float:
        """Assign distance to nearest bank branch"""
        bank_access_distance = {
            'Excellent': np.random.uniform(0.5, 3),
            'Good': np.random.uniform(2, 8),
            'Medium': np.random.uniform(5, 15),
            'Basic': np.random.uniform(8, 25),
            'Limited': np.random.uniform(15, 50)
        }
        return bank_access_distance.get(row['Bank_Access_Level'], 10)
    
    def _add_financial_goals_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add financial goals and risk appetite features"""
        # Short-term goals
        df['Short_Term_Goal'] = np.random.choice([
            'Emergency Fund', 'Loan Repayment', 'Medical Expenses', 
            'Education Fees', 'Home Repair', 'Festival Expenses'
        ], len(df))
        
        # Long-term goals
        df['Long_Term_Goal'] = np.random.choice([
            'House Purchase', 'Daughter Marriage', 'Tractor Purchase',
            'Land Purchase', 'Children Education', 'Retirement'
        ], len(df))
        
        # Time horizons
        df['Short_Term_Horizon_Months'] = np.random.randint(3, 24, len(df))
        df['Long_Term_Horizon_Years'] = np.random.randint(3, 30, len(df))
        
        # Risk tolerance
        df['Risk_Tolerance'] = df.apply(lambda row: self._assign_risk_tolerance(row), axis=1)
        
        # Investment preferences
        df['Investment_Preference'] = df.apply(lambda row: self._assign_investment_preference(row), axis=1)
        
        return df
    
    def _assign_risk_tolerance(self, row) -> str:
        """Assign risk tolerance based on multiple factors"""
        score = 0
        
        # Age factor
        if row['Age'] < 35:
            score += 2
        elif row['Age'] < 50:
            score += 1
        
        # Income factor
        if row['Monthly_Income_Primary'] > 50000:
            score += 2
        elif row['Monthly_Income_Primary'] > 25000:
            score += 1
        
        # Education factor
        if row['Education_Level'] == 'Graduate':
            score += 2
        elif row['Education_Level'] == 'Secondary':
            score += 1
        
        # Financial literacy factor
        if row['Financial_Literacy_Score'] > 7:
            score += 1
        
        if score >= 5:
            return 'High'
        elif score >= 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _assign_investment_preference(self, row) -> str:
        """Assign investment preference"""
        if row['Risk_Tolerance'] == 'Low':
            return np.random.choice(['FD', 'Savings', 'Gold', 'PPF'], p=[0.4, 0.3, 0.2, 0.1])
        elif row['Risk_Tolerance'] == 'Medium':
            return np.random.choice(['FD', 'Mutual_Funds', 'Gold', 'PPF'], p=[0.3, 0.3, 0.2, 0.2])
        else:
            return np.random.choice(['Mutual_Funds', 'Stocks', 'FD', 'Gold'], p=[0.4, 0.3, 0.2, 0.1])
    
    def _add_location_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location and seasonal factors"""
        # Generate districts within states
        df['District'] = df['State'].apply(lambda state: f"{state}_District_{np.random.randint(1, 20)}")
        
        # Crop patterns for farmers
        df['Primary_Crop'] = df.apply(lambda row: self._assign_crop_pattern(row), axis=1)
        
        # Infrastructure access
        df['Electricity_Hours_Daily'] = np.random.randint(8, 24, len(df))
        df['Water_Access'] = np.random.choice(['Tap', 'Borewell', 'Well', 'Community'], len(df))
        
        # Connectivity
        df['Internet_Access'] = df.apply(lambda row: self._assign_internet_access(row), axis=1)
        df['Road_Connectivity'] = np.random.choice(['Excellent', 'Good', 'Average', 'Poor'], len(df))
        
        return df
    
    def _assign_crop_pattern(self, row) -> str:
        """Assign crop patterns based on state and persona"""
        if 'Farmer' not in row['Persona']:
            return 'NA'
        
        state_crops = {
            'Punjab': ['Wheat', 'Rice', 'Cotton'],
            'Bihar': ['Rice', 'Wheat', 'Sugarcane'],
            'Maharashtra': ['Cotton', 'Sugarcane', 'Soybean'],
            'Karnataka': ['Rice', 'Cotton', 'Ragi'],
            'Tamil Nadu': ['Rice', 'Cotton', 'Groundnut']
        }
        
        crops = state_crops.get(row['State'], ['Rice', 'Wheat'])
        return np.random.choice(crops)
    
    def _assign_internet_access(self, row) -> bool:
        """Assign internet access based on location and demographics"""
        base_prob = 0.5
        if row['Age'] < 35:
            base_prob += 0.2
        if row['Education_Level'] in ['Secondary', 'Graduate']:
            base_prob += 0.2
        if row['Monthly_Income_Primary'] > 25000:
            base_prob += 0.1
        
        return np.random.choice([True, False], p=[min(0.9, base_prob), 1 - min(0.9, base_prob)])
    
    def _add_government_scheme_eligibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add government scheme eligibility flags"""
        # PM-Kisan eligibility
        df['PM_Kisan_Eligible'] = (df['Land_Owned_Acres'] > 0) & (df['Monthly_Income_Primary'] * 12 < 50000)
        
        # PMJJBY eligibility
        df['PMJJBY_Eligible'] = (df['Age'] >= 18) & (df['Age'] <= 50) & (df['Bank_Account_Type'] != 'None')
        
        # PMSBY eligibility
        df['PMSBY_Eligible'] = (df['Age'] >= 18) & (df['Age'] <= 70) & (df['Bank_Account_Type'] != 'None')
        
        # Atal Pension Yojana eligibility
        df['APY_Eligible'] = (df['Age'] >= 18) & (df['Age'] <= 40) & df['Occupation'].str.contains('Laborer|Worker|Farmer|Gig', na=False)
        
        # Mudra Loan eligibility
        df['Mudra_Eligible'] = df['Occupation'].str.contains('Business|Trader|Shop', na=False)
        
        # BPL status
        df['BPL_Status'] = df['Monthly_Income_Primary'] * 12 < 120000
        
        # Aadhaar linkage
        df['Aadhaar_Linked_Bank'] = np.random.choice([True, False], len(df), p=[0.85, 0.15])
        
        return df
    
    def _add_derived_financial_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived financial metrics for model training"""
        # Total monthly income
        df['Total_Monthly_Income'] = df['Monthly_Income_Primary'] + df['Monthly_Income_Secondary']
        
        # Annual income with seasonality
        df['Annual_Income_Adjusted'] = df['Total_Monthly_Income'] * 12 * df['Income_Seasonality_Factor']
        
        # Debt-to-Income ratio
        df['Debt_to_Income_Ratio'] = np.clip(df['Total_Debt_Amount'] / (df['Annual_Income_Adjusted'] + 1), 0, 5)
        
        # Total asset value
        df['Total_Asset_Value'] = (
            df['Land_Owned_Acres'] * 200000 +  # Assuming Rs 2L per acre
            df['Livestock_Value'] +
            df['Gold_Grams'] * 5000 +  # Assuming Rs 5000 per gram
            (df['Vehicle_Owned'].apply(lambda x: {'None': 0, 'Bicycle': 5000, 'Motorcycle': 80000, 'Car': 500000, 'Tractor': 800000}.get(x, 0)))
        )
        
        # Net worth
        df['Net_Worth'] = df['Total_Asset_Value'] - df['Total_Debt_Amount']
        
        # Financial stability score
        df['Financial_Stability_Score'] = self._calculate_financial_stability_score(df)
        
        # Investment capacity
        df['Monthly_Investment_Capacity'] = np.clip(
            df['Total_Monthly_Income'] * 0.2 - (df['Total_Debt_Amount'] / 12) * 0.3,
            0,
            df['Total_Monthly_Income'] * 0.5
        )
        
        return df
    
    def _calculate_financial_stability_score(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate financial stability score (1-10)"""
        score = 5.0  # Base score
        
        # Income stability factor
        stability_multiplier = {
            'Fixed': 1.5,
            'Stable': 1.3,
            'Seasonal-Stable': 1.1,
            'Variable': 0.9,
            'Seasonal': 0.7
        }
        
        score *= df['Income_Stability'].map(stability_multiplier).fillna(1.0)
        
        # Debt factor
        score -= df['Debt_to_Income_Ratio'] * 2
        
        # Asset factor
        score += np.log1p(df['Total_Asset_Value'] / 100000)
        
        # Education factor
        education_bonus = {
            'Illiterate': 0,
            'Primary': 0.5,
            'Secondary': 1.0,
            'Graduate': 1.5
        }
        score += df['Education_Level'].map(education_bonus).fillna(0)
        
        return np.clip(score, 1, 10)

if __name__ == '__main__':
    generator = EnhancedRuralDataGenerator()
    
    # Generate dataset - start with smaller size for testing
    dataset = generator.generate_ultimate_dataset(num_rows=100000)
    
    # Save dataset
    output_filename = 'enhanced_rural_financial_dataset.csv'
    dataset.to_csv(output_filename, index=False)
    
    print(f"✅ Enhanced dataset generated successfully!")
    print(f"File: {output_filename}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")
    print("\n--- Dataset Preview ---")
    print(dataset.head())
    print("\n--- Column Summary ---")
    print(dataset.dtypes)
