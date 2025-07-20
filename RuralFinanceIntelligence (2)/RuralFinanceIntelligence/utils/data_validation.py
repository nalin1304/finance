import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
from datetime import datetime, date

class DataValidator:
    """
    Comprehensive data validation for rural India financial profiles
    Ensures data quality and consistency across all parameters
    """
    
    def __init__(self):
        self.setup_logging()
        self.validation_rules = self._load_validation_rules()
        self.error_messages = []
        self.warning_messages = []
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_validation_rules(self) -> Dict:
        """Load validation rules for different parameters"""
        return {
            'age': {'min': 18, 'max': 100},
            'income_primary': {'min': 0, 'max': 10000000},
            'income_secondary': {'min': 0, 'max': 5000000},
            'dependents': {'min': 0, 'max': 15},
            'land_acres': {'min': 0, 'max': 1000},
            'debt_amount': {'min': 0, 'max': 100000000},
            'financial_literacy': {'min': 1, 'max': 10},
            'bank_distance': {'min': 0, 'max': 500},
            'valid_states': [
                'Andhra Pradesh', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka',
                'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Punjab', 'Rajasthan',
                'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'West Bengal'
            ],
            'valid_education': ['Illiterate', 'Primary', 'Secondary', 'Graduate', 'Post Graduate'],
            'valid_risk_tolerance': ['Low', 'Medium', 'High'],
            'valid_occupations': [
                'Marginal Farmer', 'Irrigated Farmer', 'Daily Wage Laborer', 'Construction Worker',
                'Small Business Owner', 'Trader', 'Shop Owner', 'Delivery Partner', 'Cab Driver',
                'Government Employee', 'Private Sector Employee', 'Teacher', 'Retired'
            ]
        }
    
    def validate_profile(self, profile: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a complete user profile
        Returns: (is_valid, errors, warnings)
        """
        self.error_messages = []
        self.warning_messages = []
        
        # Validate each category
        self._validate_demographics(profile)
        self._validate_income_employment(profile)
        self._validate_assets_liabilities(profile)
        self._validate_banking_literacy(profile)
        self._validate_financial_goals(profile)
        self._validate_location_seasonal(profile)
        self._validate_government_schemes(profile)
        
        # Cross-validation checks
        self._cross_validate_profile(profile)
        
        is_valid = len(self.error_messages) == 0
        return is_valid, self.error_messages.copy(), self.warning_messages.copy()
    
    def _validate_demographics(self, profile: Dict):
        """Validate demographic parameters"""
        # Age validation
        age = profile.get('Age')
        if age is not None:
            if not self._is_in_range(age, self.validation_rules['age']['min'], self.validation_rules['age']['max']):
                self.error_messages.append(f"Age must be between 18 and 100. Got: {age}")
        else:
            self.error_messages.append("Age is required")
        
        # Gender validation
        gender = profile.get('Gender')
        if gender not in ['Male', 'Female', 'Other']:
            self.error_messages.append(f"Invalid gender: {gender}")
        
        # Marital status validation
        marital_status = profile.get('Marital_Status')
        if marital_status not in ['Single', 'Married', 'Widowed', 'Divorced']:
            self.error_messages.append(f"Invalid marital status: {marital_status}")
        
        # Dependents validation
        dependents = profile.get('Dependents')
        if dependents is not None:
            if not self._is_in_range(dependents, 0, 15):
                self.error_messages.append(f"Dependents must be between 0 and 15. Got: {dependents}")
        
        # Education validation
        education = profile.get('Education_Level')
        if education not in self.validation_rules['valid_education']:
            self.error_messages.append(f"Invalid education level: {education}")
        
        # Occupation validation
        occupation = profile.get('Occupation')
        if occupation not in self.validation_rules['valid_occupations']:
            self.warning_messages.append(f"Uncommon occupation: {occupation}")
    
    def _validate_income_employment(self, profile: Dict):
        """Validate income and employment parameters"""
        # Primary income validation
        primary_income = profile.get('Monthly_Income_Primary', 0)
        if not self._is_in_range(primary_income, 0, 10000000):
            self.error_messages.append(f"Primary income out of range: ₹{primary_income}")
        
        # Secondary income validation
        secondary_income = profile.get('Monthly_Income_Secondary', 0)
        if not self._is_in_range(secondary_income, 0, 5000000):
            self.error_messages.append(f"Secondary income out of range: ₹{secondary_income}")
        
        # Income stability validation
        income_stability = profile.get('Income_Stability')
        valid_stability = ['Fixed', 'Stable', 'Seasonal-Stable', 'Variable', 'Seasonal']
        if income_stability not in valid_stability:
            self.error_messages.append(f"Invalid income stability: {income_stability}")
        
        # Income consistency check
        if secondary_income > primary_income:
            self.warning_messages.append("Secondary income is higher than primary income. Please verify.")
        
        # Low income warning
        total_income = primary_income + secondary_income
        if total_income < 5000:
            self.warning_messages.append("Very low income detected. Investment options may be limited.")
    
    def _validate_assets_liabilities(self, profile: Dict):
        """Validate assets and liabilities"""
        # Land ownership validation
        land_acres = profile.get('Land_Owned_Acres', 0)
        if not self._is_in_range(land_acres, 0, 1000):
            self.error_messages.append(f"Land ownership out of range: {land_acres} acres")
        
        # Livestock value validation
        livestock_value = profile.get('Livestock_Value', 0)
        if livestock_value < 0:
            self.error_messages.append("Livestock value cannot be negative")
        
        # Gold ownership validation
        gold_grams = profile.get('Gold_Grams', 0)
        if gold_grams < 0:
            self.error_messages.append("Gold ownership cannot be negative")
        if gold_grams > 10000:  # Unrealistic for rural population
            self.warning_messages.append(f"Very high gold ownership: {gold_grams} grams")
        
        # Debt validation
        debt_amount = profile.get('Total_Debt_Amount', 0)
        if debt_amount < 0:
            self.error_messages.append("Debt amount cannot be negative")
        
        debt_interest_rate = profile.get('Debt_Interest_Rate', 0)
        if debt_interest_rate < 0 or debt_interest_rate > 100:
            self.error_messages.append(f"Invalid debt interest rate: {debt_interest_rate}%")
        
        # Debt source validation
        debt_source = profile.get('Debt_Source')
        valid_sources = ['Bank', 'SHG', 'Moneylender', 'Family', 'Multiple', 'None']
        if debt_source and debt_source not in valid_sources:
            self.error_messages.append(f"Invalid debt source: {debt_source}")
        
        # High debt warning
        total_income = (profile.get('Monthly_Income_Primary', 0) + 
                       profile.get('Monthly_Income_Secondary', 0)) * 12
        if total_income > 0:
            debt_ratio = debt_amount / total_income
            if debt_ratio > 3:
                self.warning_messages.append(f"Very high debt-to-income ratio: {debt_ratio:.2f}")
    
    def _validate_banking_literacy(self, profile: Dict):
        """Validate banking and financial literacy parameters"""
        # Bank account type validation
        bank_account = profile.get('Bank_Account_Type')
        valid_accounts = ['Jan Dhan', 'Basic Savings', 'Regular Savings', 'Current', 'None']
        if bank_account not in valid_accounts:
            self.error_messages.append(f"Invalid bank account type: {bank_account}")
        
        # Bank distance validation
        bank_distance = profile.get('Bank_Distance_KM', 0)
        if not self._is_in_range(bank_distance, 0, 500):
            self.error_messages.append(f"Bank distance out of range: {bank_distance} km")
        
        # Financial literacy validation
        financial_literacy = profile.get('Financial_Literacy_Score', 5)
        if not self._is_in_range(financial_literacy, 1, 10):
            self.error_messages.append(f"Financial literacy score must be 1-10. Got: {financial_literacy}")
        
        # Digital literacy validation
        digital_literacy = profile.get('Digital_Literacy')
        valid_digital = ['Low', 'Low-Medium', 'Medium', 'Medium-High', 'High']
        if digital_literacy not in valid_digital:
            self.error_messages.append(f"Invalid digital literacy: {digital_literacy}")
        
        # Consistency checks
        smartphone = profile.get('Smartphone_Access', False)
        internet = profile.get('Internet_Access', False)
        upi_usage = profile.get('UPI_Usage', False)
        
        if upi_usage and not smartphone:
            self.warning_messages.append("UPI usage without smartphone access seems inconsistent")
        
        if internet and not smartphone:
            self.warning_messages.append("Internet access without smartphone is uncommon in rural areas")
    
    def _validate_financial_goals(self, profile: Dict):
        """Validate financial goals and risk parameters"""
        # Risk tolerance validation
        risk_tolerance = profile.get('Risk_Tolerance')
        if risk_tolerance not in self.validation_rules['valid_risk_tolerance']:
            self.error_messages.append(f"Invalid risk tolerance: {risk_tolerance}")
        
        # Time horizon validation
        short_horizon = profile.get('Short_Term_Horizon_Months', 12)
        if not self._is_in_range(short_horizon, 1, 60):
            self.error_messages.append(f"Short term horizon out of range: {short_horizon} months")
        
        long_horizon = profile.get('Long_Term_Horizon_Years', 10)
        if not self._is_in_range(long_horizon, 1, 50):
            self.error_messages.append(f"Long term horizon out of range: {long_horizon} years")
        
        # Goal consistency
        if short_horizon >= long_horizon * 12:
            self.warning_messages.append("Short term horizon is longer than long term horizon")
        
        # Investment preference validation
        investment_pref = profile.get('Investment_Preference')
        valid_prefs = ['FD', 'Savings', 'Gold', 'PPF', 'Mutual_Funds', 'Stocks', 'Real_Estate']
        if investment_pref and investment_pref not in valid_prefs:
            self.warning_messages.append(f"Uncommon investment preference: {investment_pref}")
    
    def _validate_location_seasonal(self, profile: Dict):
        """Validate location and seasonal factors"""
        # State validation
        state = profile.get('State')
        if state not in self.validation_rules['valid_states']:
            self.error_messages.append(f"Invalid state: {state}")
        
        # Electricity hours validation
        electricity_hours = profile.get('Electricity_Hours_Daily', 18)
        if not self._is_in_range(electricity_hours, 0, 24):
            self.error_messages.append(f"Electricity hours must be 0-24. Got: {electricity_hours}")
        
        # Water access validation
        water_access = profile.get('Water_Access')
        valid_water = ['Tap', 'Borewell', 'Well', 'Community', 'None']
        if water_access and water_access not in valid_water:
            self.error_messages.append(f"Invalid water access: {water_access}")
        
        # Road connectivity validation
        road_connectivity = profile.get('Road_Connectivity')
        valid_roads = ['Excellent', 'Good', 'Average', 'Poor']
        if road_connectivity and road_connectivity not in valid_roads:
            self.error_messages.append(f"Invalid road connectivity: {road_connectivity}")
        
        # Crop validation for farmers
        occupation = profile.get('Occupation', '')
        primary_crop = profile.get('Primary_Crop', 'NA')
        
        if 'Farmer' in occupation and primary_crop == 'NA':
            self.warning_messages.append("Farmer occupation but no crop specified")
        elif 'Farmer' not in occupation and primary_crop != 'NA':
            self.warning_messages.append("Crop specified but not a farmer")
    
    def _validate_government_schemes(self, profile: Dict):
        """Validate government scheme eligibility"""
        age = profile.get('Age', 30)
        annual_income = (profile.get('Monthly_Income_Primary', 0) + 
                        profile.get('Monthly_Income_Secondary', 0)) * 12
        land_acres = profile.get('Land_Owned_Acres', 0)
        
        # PM-Kisan eligibility validation
        pm_kisan_eligible = profile.get('PM_Kisan_Eligible', False)
        if pm_kisan_eligible and (land_acres == 0 or annual_income > 200000):
            self.warning_messages.append("PM-Kisan eligibility seems inconsistent with land/income")
        
        # PMJJBY eligibility validation
        pmjjby_eligible = profile.get('PMJJBY_Eligible', False)
        if pmjjby_eligible and (age < 18 or age > 50):
            self.error_messages.append("PMJJBY eligibility age criteria not met")
        
        # PMSBY eligibility validation
        pmsby_eligible = profile.get('PMSBY_Eligible', False)
        if pmsby_eligible and (age < 18 or age > 70):
            self.error_messages.append("PMSBY eligibility age criteria not met")
        
        # APY eligibility validation
        apy_eligible = profile.get('APY_Eligible', False)
        if apy_eligible and (age < 18 or age > 40):
            self.error_messages.append("APY eligibility age criteria not met")
        
        # BPL status validation
        bpl_status = profile.get('BPL_Status', False)
        if not bpl_status and annual_income < 50000:
            self.warning_messages.append("Low income but not marked as BPL")
        elif bpl_status and annual_income > 200000:
            self.warning_messages.append("High income but marked as BPL")
    
    def _cross_validate_profile(self, profile: Dict):
        """Perform cross-validation checks across parameters"""
        age = profile.get('Age', 30)
        education = profile.get('Education_Level', 'Primary')
        occupation = profile.get('Occupation', '')
        digital_literacy = profile.get('Digital_Literacy', 'Low')
        financial_literacy = profile.get('Financial_Literacy_Score', 5)
        
        # Education-occupation consistency
        if education == 'Graduate' and 'Laborer' in occupation:
            self.warning_messages.append("Graduate education with laborer occupation seems inconsistent")
        
        if education == 'Illiterate' and 'Employee' in occupation:
            self.warning_messages.append("Illiterate with formal employment seems inconsistent")
        
        # Age-occupation consistency
        if age > 65 and occupation != 'Retired':
            self.warning_messages.append("Advanced age but not retired")
        
        if age < 25 and occupation == 'Retired':
            self.error_messages.append("Too young to be retired")
        
        # Digital literacy-age consistency
        if age > 60 and digital_literacy == 'High':
            self.warning_messages.append("High digital literacy at advanced age is uncommon")
        
        if age < 30 and digital_literacy == 'Low':
            self.warning_messages.append("Low digital literacy in young age is uncommon")
        
        # Financial literacy-education consistency
        if education == 'Graduate' and financial_literacy < 5:
            self.warning_messages.append("Graduate education with low financial literacy")
        
        if education == 'Illiterate' and financial_literacy > 7:
            self.warning_messages.append("High financial literacy despite illiteracy")
        
        # Income-asset consistency
        total_income = (profile.get('Monthly_Income_Primary', 0) + 
                       profile.get('Monthly_Income_Secondary', 0)) * 12
        total_assets = profile.get('Total_Asset_Value', 0)
        
        if total_income > 500000 and total_assets < 100000:
            self.warning_messages.append("High income but low assets seems inconsistent")
        
        if total_income < 100000 and total_assets > 1000000:
            self.warning_messages.append("Low income but high assets seems inconsistent")
    
    def _is_in_range(self, value, min_val, max_val) -> bool:
        """Check if value is in valid range"""
        return min_val <= value <= max_val
    
    def validate_investment_amount(self, amount: float, profile: Dict) -> Tuple[bool, List[str]]:
        """Validate investment amount against user profile"""
        errors = []
        
        if amount <= 0:
            errors.append("Investment amount must be positive")
            return False, errors
        
        total_monthly_income = (profile.get('Monthly_Income_Primary', 0) + 
                               profile.get('Monthly_Income_Secondary', 0))
        
        # Check if investment amount is reasonable
        if amount > total_monthly_income * 60:  # More than 5 years of income
            errors.append("Investment amount seems very high compared to income")
        
        # Check minimum investment thresholds
        if amount < 500:
            errors.append("Minimum investment amount should be ₹500")
        
        # Check debt situation
        debt_ratio = profile.get('Debt_to_Income_Ratio', 0)
        if debt_ratio > 2 and amount > total_monthly_income * 3:
            errors.append("High debt situation - consider reducing investment amount")
        
        return len(errors) == 0, errors
    
    def get_data_quality_score(self, profile: Dict) -> float:
        """Calculate data quality score for a profile"""
        is_valid, errors, warnings = self.validate_profile(profile)
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for errors and warnings
        score -= len(errors) * 10  # 10 points per error
        score -= len(warnings) * 2  # 2 points per warning
        
        # Check completeness
        required_fields = [
            'Age', 'Gender', 'Education_Level', 'Occupation',
            'Monthly_Income_Primary', 'Risk_Tolerance', 'State'
        ]
        
        missing_fields = sum(1 for field in required_fields if not profile.get(field))
        score -= missing_fields * 5  # 5 points per missing required field
        
        return max(0.0, min(100.0, score))

if __name__ == '__main__':
    # Test the validator
    validator = DataValidator()
    
    # Test profile
    test_profile = {
        'Age': 35,
        'Gender': 'Male',
        'Marital_Status': 'Married',
        'Education_Level': 'Secondary',
        'Occupation': 'Small Business Owner',
        'Monthly_Income_Primary': 25000,
        'Monthly_Income_Secondary': 5000,
        'Risk_Tolerance': 'Medium',
        'State': 'Maharashtra',
        'Financial_Literacy_Score': 6,
        'Digital_Literacy': 'Medium',
        'Land_Owned_Acres': 2.0,
        'Total_Debt_Amount': 100000,
        'Debt_Interest_Rate': 12.0
    }
    
    is_valid, errors, warnings = validator.validate_profile(test_profile)
    quality_score = validator.get_data_quality_score(test_profile)
    
    print(f"Profile Valid: {is_valid}")
    print(f"Quality Score: {quality_score:.1f}/100")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
