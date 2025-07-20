import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sqlite3

from market_data_fetcher import MarketDataFetcher
from utils.financial_calculations import FinancialCalculator
from utils.data_validation import DataValidator
from config.investment_products import INVESTMENT_PRODUCTS
from config.government_schemes import GOVERNMENT_SCHEMES

class AdvancedRecommendationEngine:
    """
    Advanced recommendation engine for rural India financial advisory
    Generates specific product recommendations with exact allocations
    """
    
    def __init__(self, model_path: str = None):
        self.setup_logging()
        self.market_fetcher = MarketDataFetcher()
        self.financial_calc = FinancialCalculator()
        self.validator = DataValidator()
        
        # Load ML models if available
        self.models = None
        self.preprocessor = None
        self.feature_names = None
        self.target_names = None
        
        if model_path:
            self.load_models(model_path)
        
        # Cache for market data
        self.market_data_cache = None
        self.cache_timestamp = None
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_models(self, model_path: str):
        """Load trained ML models"""
        try:
            model_data = joblib.load(model_path)
            self.models = model_data['models']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            self.target_names = model_data['target_names']
            self.logger.info(f"Successfully loaded models from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_fresh_market_data(self) -> Dict:
        """Get fresh market data with caching"""
        # Check if cache is still valid (30 minutes)
        if (self.market_data_cache is None or 
            self.cache_timestamp is None or 
            (datetime.now() - self.cache_timestamp).seconds > 1800):
            
            self.logger.info("Fetching fresh market data...")
            self.market_data_cache = self.market_fetcher.fetch_all_market_data()
            self.cache_timestamp = datetime.now()
        
        return self.market_data_cache
    
    def predict_portfolio_allocation(self, user_profile: Dict) -> Dict:
        """
        Predict portfolio allocation using ML models
        """
        if not self.models or not self.preprocessor:
            # Fallback to rule-based allocation
            return self._rule_based_allocation(user_profile)
        
        try:
            # Prepare user data for prediction
            user_df = pd.DataFrame([user_profile])
            
            # Add derived features
            user_df = self._add_derived_features(user_df)
            
            # Select only the features used for training
            feature_subset = [f for f in self.feature_names if f in user_df.columns]
            X = user_df[feature_subset]
            
            # Fill missing features with defaults
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0
            
            # Reorder columns to match training
            X = X[self.feature_names]
            
            # Transform features
            X_transformed = self.preprocessor.transform(X)
            
            # Predict allocations using best models
            allocations = {}
            for target in self.target_names:
                if target in self.models:
                    # Use the best performing model for each target
                    best_model_name = max(self.models[target].keys(), 
                                        key=lambda k: self.models[target][k])
                    model = self.models[target][best_model_name]
                    
                    prediction = model.predict(X_transformed)[0]
                    allocations[target] = max(0, min(1, prediction))  # Clip to [0,1]
            
            # Normalize allocations to sum to 1
            total = sum(allocations.values())
            if total > 0:
                allocations = {k: v/total for k, v in allocations.items()}
            else:
                # Fallback to rule-based if predictions fail
                return self._rule_based_allocation(user_profile)
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            return self._rule_based_allocation(user_profile)
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features for ML prediction"""
        # Risk capacity score
        df['Risk_Capacity_Score'] = self._calculate_risk_capacity(df.iloc[0])
        
        # Investment readiness score
        df['Investment_Readiness_Score'] = self._calculate_investment_readiness(df.iloc[0])
        
        # Accessibility score
        df['Accessibility_Score'] = self._calculate_accessibility(df.iloc[0])
        
        return df
    
    def _calculate_risk_capacity(self, profile: Dict) -> float:
        """Calculate risk capacity score"""
        score = 0
        
        # Income factor
        monthly_income = profile.get('Monthly_Income_Primary', 0) + profile.get('Monthly_Income_Secondary', 0)
        if monthly_income > 50000:
            score += 3
        elif monthly_income > 25000:
            score += 2
        elif monthly_income > 15000:
            score += 1
        
        # Age factor
        age = profile.get('Age', 40)
        if age < 35:
            score += 2
        elif age < 50:
            score += 1
        
        # Debt factor
        debt_ratio = profile.get('Debt_to_Income_Ratio', 0.5)
        if debt_ratio < 0.3:
            score += 2
        elif debt_ratio < 0.5:
            score += 1
        
        # Asset factor
        assets = profile.get('Total_Asset_Value', 0)
        if assets > 500000:
            score += 2
        elif assets > 200000:
            score += 1
        
        return min(10, score)
    
    def _calculate_investment_readiness(self, profile: Dict) -> float:
        """Calculate investment readiness score"""
        score = 0
        
        # Financial literacy
        literacy = profile.get('Financial_Literacy_Score', 5)
        score += literacy
        
        # Digital literacy
        digital = profile.get('Digital_Literacy', 'Low')
        digital_scores = {'High': 3, 'Medium-High': 2.5, 'Medium': 2, 'Low-Medium': 1.5, 'Low': 1}
        score += digital_scores.get(digital, 1)
        
        # Banking access
        bank_access = profile.get('Bank_Access_Level', 'Basic')
        bank_scores = {'Excellent': 2, 'Good': 1.5, 'Medium': 1, 'Basic': 0.5, 'Limited': 0}
        score += bank_scores.get(bank_access, 0.5)
        
        return min(10, score)
    
    def _calculate_accessibility(self, profile: Dict) -> float:
        """Calculate accessibility score"""
        score = 5  # Base score
        
        # Bank distance
        distance = profile.get('Bank_Distance_KM', 10)
        if distance < 5:
            score += 2
        elif distance < 15:
            score += 1
        else:
            score -= 1
        
        # Digital access
        if profile.get('Smartphone_Access', False):
            score += 1
        if profile.get('Internet_Access', False):
            score += 1
        if profile.get('UPI_Usage', False):
            score += 1
        
        return min(10, max(0, score))
    
    def _rule_based_allocation(self, user_profile: Dict) -> Dict:
        """
        Fallback rule-based portfolio allocation
        """
        risk_tolerance = user_profile.get('Risk_Tolerance', 'Low')
        age = user_profile.get('Age', 40)
        monthly_income = user_profile.get('Monthly_Income_Primary', 0) + user_profile.get('Monthly_Income_Secondary', 0)
        
        # Base allocation based on risk tolerance
        if risk_tolerance == 'Low':
            allocation = {
                'Equity_Large_Cap': 0.05,
                'Equity_Mid_Cap': 0.0,
                'ELSS': 0.05,
                'Debt_Govt_Scheme': 0.40,
                'Debt_FD': 0.35,
                'PPF': 0.10,
                'Gold': 0.05,
                'Cash': 0.0
            }
        elif risk_tolerance == 'Medium':
            allocation = {
                'Equity_Large_Cap': 0.25,
                'Equity_Mid_Cap': 0.10,
                'ELSS': 0.10,
                'Debt_Govt_Scheme': 0.20,
                'Debt_FD': 0.20,
                'PPF': 0.10,
                'Gold': 0.05,
                'Cash': 0.0
            }
        else:  # High
            allocation = {
                'Equity_Large_Cap': 0.40,
                'Equity_Mid_Cap': 0.20,
                'ELSS': 0.15,
                'Debt_Govt_Scheme': 0.10,
                'Debt_FD': 0.10,
                'PPF': 0.0,
                'Gold': 0.05,
                'Cash': 0.0
            }
        
        # Adjust based on age
        if age > 55:
            # Reduce equity, increase debt
            equity_reduction = 0.15
            allocation['Equity_Large_Cap'] = max(0, allocation['Equity_Large_Cap'] - equity_reduction)
            allocation['Debt_FD'] += equity_reduction * 0.7
            allocation['PPF'] += equity_reduction * 0.3
        
        # Adjust based on income
        if monthly_income < 15000:
            # Increase government schemes and traditional instruments
            allocation['Debt_Govt_Scheme'] += 0.1
            allocation['Gold'] += 0.05
            allocation['Equity_Large_Cap'] = max(0, allocation['Equity_Large_Cap'] - 0.15)
        
        # Normalize
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v/total for k, v in allocation.items()}
        
        return allocation
    
    def generate_specific_recommendations(self, user_profile: Dict, investment_amount: float) -> Dict:
        """
        Generate specific product recommendations with exact allocations
        """
        self.logger.info(f"Generating recommendations for investment amount: ₹{investment_amount:,.0f}")
        
        # Get portfolio allocation
        allocation = self.predict_portfolio_allocation(user_profile)
        
        # Get fresh market data
        market_data = self.get_fresh_market_data()
        
        # Generate specific product recommendations
        recommendations = {
            'total_investment': investment_amount,
            'allocations': allocation,
            'specific_products': [],
            'summary': {
                'expected_annual_return': 0,
                'risk_level': user_profile.get('Risk_Tolerance', 'Medium'),
                'tax_benefits': 0,
                'liquidity_score': 0
            },
            'eligibility_alerts': [],
            'accessibility_warnings': []
        }
        
        # Generate specific product recommendations for each allocation
        for asset_class, percentage in allocation.items():
            if percentage > 0:
                amount = investment_amount * percentage
                products = self._get_specific_products(asset_class, amount, user_profile, market_data)
                recommendations['specific_products'].extend(products)
        
        # Calculate summary metrics
        recommendations['summary'] = self._calculate_portfolio_summary(
            recommendations['specific_products'], user_profile
        )
        
        # Check eligibility and accessibility
        recommendations['eligibility_alerts'] = self._check_eligibility(user_profile)
        recommendations['accessibility_warnings'] = self._check_accessibility(user_profile)
        
        # Sort recommendations by amount (descending)
        recommendations['specific_products'].sort(key=lambda x: x['amount'], reverse=True)
        
        return recommendations
    
    def _get_specific_products(self, asset_class: str, amount: float, user_profile: Dict, market_data: Dict) -> List[Dict]:
        """Get specific product recommendations for an asset class"""
        products = []
        
        if asset_class == 'Equity_Large_Cap':
            products.extend(self._get_equity_recommendations(amount, user_profile, market_data, 'Large_Cap'))
        
        elif asset_class == 'Equity_Mid_Cap':
            products.extend(self._get_equity_recommendations(amount, user_profile, market_data, 'Mid_Cap'))
        
        elif asset_class == 'ELSS':
            products.extend(self._get_elss_recommendations(amount, user_profile, market_data))
        
        elif asset_class == 'Debt_FD':
            products.extend(self._get_fd_recommendations(amount, user_profile, market_data))
        
        elif asset_class == 'Debt_Govt_Scheme':
            products.extend(self._get_govt_scheme_recommendations(amount, user_profile, market_data))
        
        elif asset_class == 'PPF':
            products.extend(self._get_ppf_recommendations(amount, user_profile))
        
        elif asset_class == 'Gold':
            products.extend(self._get_gold_recommendations(amount, user_profile, market_data))
        
        return products
    
    def _get_equity_recommendations(self, amount: float, user_profile: Dict, market_data: Dict, cap_type: str) -> List[Dict]:
        """Get specific equity stock recommendations"""
        products = []
        
        # Use real stock data if available
        stocks = market_data.get('stocks', {})
        if not stocks:
            return products
        
        # Filter stocks by market cap (simplified approach)
        filtered_stocks = []
        for symbol, data in stocks.items():
            if isinstance(data, dict):
                market_cap = data.get('market_cap', 0)
                if cap_type == 'Large_Cap' and market_cap > 1000000000000:  # > 1T market cap
                    filtered_stocks.append((symbol, data))
                elif cap_type == 'Mid_Cap' and 100000000000 < market_cap <= 1000000000000:  # 100B-1T
                    filtered_stocks.append((symbol, data))
        
        # Sort by performance and select top 3
        filtered_stocks.sort(key=lambda x: x[1].get('change_percent', 0), reverse=True)
        selected_stocks = filtered_stocks[:3]
        
        # Distribute amount across selected stocks
        per_stock_amount = amount / len(selected_stocks) if selected_stocks else amount
        
        for symbol, data in selected_stocks:
            price = data.get('current_price', 100)
            shares = int(per_stock_amount / price)
            actual_amount = shares * price
            
            if shares > 0:
                products.append({
                    'type': 'Equity',
                    'category': cap_type,
                    'name': data.get('company_name', symbol.replace('.NS', '')),
                    'symbol': symbol,
                    'price_per_share': price,
                    'shares': shares,
                    'amount': actual_amount,
                    'allocation_percentage': actual_amount / amount * 100 if amount > 0 else 0,
                    'expected_return': 12 if cap_type == 'Large_Cap' else 15,
                    'risk_level': 'Medium' if cap_type == 'Large_Cap' else 'High',
                    'liquidity': 'High',
                    'tax_implication': '10% LTCG after 1 year',
                    'min_investment': price,
                    'sector': data.get('sector', 'Unknown')
                })
        
        return products
    
    def _get_elss_recommendations(self, amount: float, user_profile: Dict, market_data: Dict) -> List[Dict]:
        """Get ELSS fund recommendations"""
        products = []
        
        # Get ELSS funds from market data
        elss_funds = market_data.get('mutual_funds', {}).get('ELSS', [])
        
        if not elss_funds:
            # Fallback to configured ELSS funds
            elss_funds = [
                {
                    'scheme_name': 'Axis Long Term Equity Fund - Direct Plan - Growth',
                    'nav': 65.50,
                    'returns_3y': 12.5,
                    'expense_ratio': 0.8,
                    'min_investment': 1000
                }
            ]
        
        # Select top 2 performing ELSS funds
        elss_funds.sort(key=lambda x: x.get('returns_3y', 0), reverse=True)
        selected_funds = elss_funds[:2]
        
        per_fund_amount = amount / len(selected_funds) if selected_funds else amount
        
        for fund in selected_funds:
            nav = fund.get('nav', 50)
            units = per_fund_amount / nav
            actual_amount = units * nav
            
            products.append({
                'type': 'ELSS Mutual Fund',
                'category': 'Tax Saving',
                'name': fund.get('scheme_name', 'ELSS Fund'),
                'nav': nav,
                'units': round(units, 3),
                'amount': actual_amount,
                'allocation_percentage': actual_amount / amount * 100 if amount > 0 else 0,
                'expected_return': fund.get('returns_3y', 12),
                'risk_level': 'High',
                'liquidity': 'Low (3 year lock-in)',
                'tax_benefit': '80C deduction up to ₹1.5L',
                'tax_implication': '10% LTCG after 1 year',
                'min_investment': fund.get('min_investment', 1000),
                'expense_ratio': fund.get('expense_ratio', 1.0),
                'lock_in_period': '3 years'
            })
        
        return products
    
    def _get_fd_recommendations(self, amount: float, user_profile: Dict, market_data: Dict) -> List[Dict]:
        """Get Fixed Deposit recommendations"""
        products = []
        
        # Get FD rates from market data
        fd_data = market_data.get('fd_rates', {})
        
        if not fd_data:
            return products
        
        # Find best FD rates
        best_fds = []
        for bank, fd_list in fd_data.items():
            if isinstance(fd_list, list):
                for fd_info in fd_list:
                    if fd_info.get('tenure_months') in [12, 24, 36]:  # Prefer 1-3 year tenures
                        best_fds.append((bank, fd_info))
        
        # Sort by interest rate
        best_fds.sort(key=lambda x: x[1].get('rate', 0), reverse=True)
        
        # Select top 2 FDs
        selected_fds = best_fds[:2]
        per_fd_amount = amount / len(selected_fds) if selected_fds else amount
        
        # Check if user is senior citizen
        is_senior = user_profile.get('Age', 40) >= 60
        
        for bank, fd_info in selected_fds:
            rate = fd_info.get('senior_citizen_rate' if is_senior else 'rate', 7.0)
            tenure_months = fd_info.get('tenure_months', 12)
            min_amount = fd_info.get('min_amount', 1000)
            
            actual_amount = max(per_fd_amount, min_amount)
            maturity_amount = self.financial_calc.calculate_fd_maturity(actual_amount, rate, tenure_months)
            
            products.append({
                'type': 'Fixed Deposit',
                'category': 'Debt',
                'name': f"{bank} FD - {tenure_months} months",
                'bank': bank,
                'principal': actual_amount,
                'amount': actual_amount,
                'allocation_percentage': actual_amount / amount * 100 if amount > 0 else 0,
                'interest_rate': rate,
                'tenure_months': tenure_months,
                'maturity_amount': maturity_amount,
                'expected_return': rate,
                'risk_level': 'Very Low',
                'liquidity': 'Low (penalty on premature withdrawal)',
                'tax_implication': 'Interest taxable as per income slab',
                'min_investment': min_amount,
                'deposit_insurance': 'DICGC insured up to ₹5 lakh',
                'bank_type': fd_info.get('bank_type', 'Unknown')
            })
        
        return products
    
    def _get_govt_scheme_recommendations(self, amount: float, user_profile: Dict, market_data: Dict) -> List[Dict]:
        """Get government scheme recommendations"""
        products = []
        
        # Get government schemes data
        govt_schemes = market_data.get('government_schemes', {})
        
        # Prioritize NSC and other government schemes
        recommended_schemes = ['NSC', 'NPS']
        
        for scheme_name in recommended_schemes:
            if scheme_name in govt_schemes:
                scheme = govt_schemes[scheme_name]
                
                min_investment = scheme.get('min_investment', 1000)
                max_investment = scheme.get('max_investment')
                
                # Calculate investment amount
                scheme_amount = amount / len(recommended_schemes)
                
                if max_investment:
                    scheme_amount = min(scheme_amount, max_investment)
                
                scheme_amount = max(scheme_amount, min_investment)
                
                if scheme_name == 'NSC':
                    maturity_amount = self.financial_calc.calculate_compound_interest(
                        scheme_amount, scheme.get('interest_rate', 6.8), 5
                    )
                    
                    products.append({
                        'type': 'Government Scheme',
                        'category': 'Tax Saving Debt',
                        'name': scheme.get('full_name', 'National Savings Certificate'),
                        'scheme_code': 'NSC',
                        'investment_amount': scheme_amount,
                        'amount': scheme_amount,
                        'allocation_percentage': scheme_amount / amount * 100 if amount > 0 else 0,
                        'interest_rate': scheme.get('interest_rate', 6.8),
                        'tenure_years': scheme.get('lock_in_period_years', 5),
                        'maturity_amount': maturity_amount,
                        'expected_return': scheme.get('interest_rate', 6.8),
                        'risk_level': 'Very Low',
                        'liquidity': 'Very Low (No premature withdrawal)',
                        'tax_benefit': scheme.get('tax_benefit', '80C deduction'),
                        'tax_implication': 'Interest reinvestment eligible for 80C',
                        'min_investment': min_investment,
                        'eligibility': scheme.get('eligibility', 'All citizens'),
                        'government_guarantee': True
                    })
        
        return products
    
    def _get_ppf_recommendations(self, amount: float, user_profile: Dict) -> List[Dict]:
        """Get PPF recommendations"""
        products = []
        
        # PPF has a limit of ₹1.5 lakh per year
        max_ppf = 150000
        ppf_amount = min(amount, max_ppf)
        
        if ppf_amount >= 500:  # Minimum PPF investment
            maturity_amount = self.financial_calc.calculate_compound_interest(ppf_amount, 7.1, 15)
            
            products.append({
                'type': 'Government Scheme',
                'category': 'Tax Saving',
                'name': 'Public Provident Fund (PPF)',
                'scheme_code': 'PPF',
                'investment_amount': ppf_amount,
                'amount': ppf_amount,
                'allocation_percentage': ppf_amount / amount * 100 if amount > 0 else 0,
                'interest_rate': 7.1,
                'tenure_years': 15,
                'maturity_amount': maturity_amount,
                'expected_return': 7.1,
                'risk_level': 'Very Low',
                'liquidity': 'Low (Partial withdrawal after 7 years)',
                'tax_benefit': 'EEE - Triple tax exemption',
                'tax_implication': 'No tax on interest or maturity',
                'min_investment': 500,
                'max_annual_investment': 150000,
                'government_guarantee': True,
                'extension_option': 'Available after 15 years'
            })
        
        return products
    
    def _get_gold_recommendations(self, amount: float, user_profile: Dict, market_data: Dict) -> List[Dict]:
        """Get gold investment recommendations"""
        products = []
        
        # Recommend Gold ETF for better liquidity and lower storage costs
        current_gold_price = 5500  # Approximate price per gram
        
        # Gold ETF recommendation
        etf_amount = amount * 0.7  # 70% in Gold ETF
        digital_gold_amount = amount * 0.3  # 30% in Digital Gold
        
        if etf_amount >= 1000:
            products.append({
                'type': 'Gold ETF',
                'category': 'Commodity',
                'name': 'HDFC Gold ETF or SBI Gold ETF',
                'investment_amount': etf_amount,
                'amount': etf_amount,
                'allocation_percentage': etf_amount / amount * 100 if amount > 0 else 0,
                'gold_equivalent_grams': etf_amount / current_gold_price,
                'expected_return': 8,  # Historical gold returns
                'risk_level': 'Medium',
                'liquidity': 'High (Can trade on stock exchange)',
                'tax_implication': '20% LTCG with indexation after 3 years',
                'min_investment': 1000,
                'storage_cost': 'No physical storage required',
                'purity': '99.5% gold backed'
            })
        
        if digital_gold_amount >= 500:
            products.append({
                'type': 'Digital Gold',
                'category': 'Commodity',
                'name': 'Digital Gold (Paytm Gold/Google Pay Gold)',
                'investment_amount': digital_gold_amount,
                'amount': digital_gold_amount,
                'allocation_percentage': digital_gold_amount / amount * 100 if amount > 0 else 0,
                'gold_equivalent_grams': digital_gold_amount / current_gold_price,
                'expected_return': 8,
                'risk_level': 'Medium',
                'liquidity': 'High (Instant buy/sell)',
                'tax_implication': '20% LTCG with indexation after 3 years',
                'min_investment': 1,
                'storage_cost': 'Digital storage included',
                'purity': '24K gold'
            })
        
        return products
    
    def _calculate_portfolio_summary(self, products: List[Dict], user_profile: Dict) -> Dict:
        """Calculate portfolio summary metrics"""
        total_amount = sum(p['amount'] for p in products)
        weighted_return = sum(p['amount'] * p.get('expected_return', 0) for p in products) / total_amount if total_amount > 0 else 0
        
        # Calculate tax benefits
        tax_benefits = 0
        for product in products:
            if '80C' in product.get('tax_benefit', ''):
                tax_benefits += min(product['amount'], 150000)  # 80C limit
            elif 'EEE' in product.get('tax_benefit', ''):
                tax_benefits += min(product['amount'], 150000)
        
        # Calculate liquidity score (weighted average)
        liquidity_scores = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4, 'Very High': 5}
        weighted_liquidity = sum(
            p['amount'] * liquidity_scores.get(p.get('liquidity', 'Medium').split(' (')[0], 3) 
            for p in products
        ) / total_amount if total_amount > 0 else 3
        
        return {
            'expected_annual_return': round(weighted_return, 2),
            'risk_level': user_profile.get('Risk_Tolerance', 'Medium'),
            'tax_benefits': round(tax_benefits, 0),
            'liquidity_score': round(weighted_liquidity, 1),
            'total_products': len(products),
            'diversification_score': min(10, len(set(p['category'] for p in products)) * 2)
        }
    
    def _check_eligibility(self, user_profile: Dict) -> List[str]:
        """Check eligibility for various schemes and products"""
        alerts = []
        
        age = user_profile.get('Age', 30)
        annual_income = (user_profile.get('Monthly_Income_Primary', 0) + 
                        user_profile.get('Monthly_Income_Secondary', 0)) * 12
        
        # Check PM-Kisan eligibility
        if (user_profile.get('Land_Owned_Acres', 0) > 0 and 
            annual_income < 200000 and 
            not user_profile.get('PM_Kisan_Eligible', False)):
            alerts.append("You may be eligible for PM-Kisan scheme (₹6,000/year). Please verify at local agriculture office.")
        
        # Check PMSBY eligibility
        if (18 <= age <= 70 and 
            user_profile.get('Bank_Account_Type') != 'None' and
            not user_profile.get('PMSBY_Eligible', False)):
            alerts.append("Consider PMSBY accident insurance for just ₹12/year covering ₹2 lakh.")
        
        # Check APY eligibility
        if (18 <= age <= 40 and 
            not user_profile.get('APY_Eligible', False)):
            alerts.append("You're eligible for Atal Pension Yojana for guaranteed pension of ₹1000-5000/month.")
        
        # Check senior citizen benefits
        if age >= 60:
            alerts.append("As a senior citizen, you get higher FD rates and additional tax benefits.")
        
        return alerts
    
    def _check_accessibility(self, user_profile: Dict) -> List[str]:
        """Check accessibility constraints and provide warnings"""
        warnings = []
        
        # Bank distance warning
        bank_distance = user_profile.get('Bank_Distance_KM', 10)
        if bank_distance > 15:
            warnings.append(f"Nearest bank is {bank_distance}km away. Consider digital banking options or SHG participation.")
        
        # Digital literacy warning
        if user_profile.get('Digital_Literacy', 'Low') == 'Low':
            warnings.append("Limited digital access may restrict online investment options. Consider visiting bank branches.")
        
        # Smartphone access warning
        if not user_profile.get('Smartphone_Access', False):
            warnings.append("Without smartphone access, mobile banking and UPI services won't be available.")
        
        # Income threshold warnings
        monthly_income = user_profile.get('Monthly_Income_Primary', 0)
        if monthly_income < 10000:
            warnings.append("Consider starting with small amounts and government schemes before equity investments.")
        
        # Education warning
        if user_profile.get('Education_Level') in ['Illiterate', 'Primary']:
            warnings.append("Consider seeking financial guidance from bank representatives or local financial advisors.")
        
        return warnings

if __name__ == '__main__':
    # Test the recommendation engine
    engine = AdvancedRecommendationEngine()
    
    # Sample user profile
    test_profile = {
        'Age': 35,
        'Monthly_Income_Primary': 25000,
        'Monthly_Income_Secondary': 5000,
        'Risk_Tolerance': 'Medium',
        'Financial_Literacy_Score': 6,
        'Digital_Literacy': 'Medium',
        'Bank_Access_Level': 'Good',
        'Bank_Distance_KM': 5,
        'Smartphone_Access': True,
        'Internet_Access': True,
        'UPI_Usage': True,
        'Education_Level': 'Secondary',
        'Land_Owned_Acres': 2.0,
        'Total_Asset_Value': 300000,
        'Debt_to_Income_Ratio': 0.3
    }
    
    recommendations = engine.generate_specific_recommendations(test_profile, 100000)
    
    print("🎯 FINANCIAL RECOMMENDATIONS")
    print("=" * 50)
    print(f"Total Investment: ₹{recommendations['total_investment']:,.0f}")
    print(f"Expected Annual Return: {recommendations['summary']['expected_annual_return']:.1f}%")
    print(f"Risk Level: {recommendations['summary']['risk_level']}")
    print(f"Tax Benefits: ₹{recommendations['summary']['tax_benefits']:,.0f}")
    
    print("\n📊 SPECIFIC PRODUCT RECOMMENDATIONS:")
    for i, product in enumerate(recommendations['specific_products'], 1):
        print(f"\n{i}. {product['name']}")
        print(f"   Amount: ₹{product['amount']:,.0f} ({product['allocation_percentage']:.1f}%)")
        print(f"   Expected Return: {product['expected_return']:.1f}%")
        print(f"   Risk: {product['risk_level']}")
