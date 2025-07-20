import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math
from datetime import datetime, timedelta
import logging

class FinancialCalculator:
    """
    Comprehensive financial calculations for rural India investment scenarios
    Includes complex calculations for various financial products and tax implications
    """
    
    def __init__(self):
        self.setup_logging()
        self.current_tax_rates = self._load_current_tax_rates()
        self.current_interest_rates = self._load_current_interest_rates()
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_current_tax_rates(self) -> Dict:
        """Load current tax rates for 2025"""
        return {
            'income_tax_slabs_old': [
                (250000, 0.0),
                (500000, 0.05),
                (1000000, 0.20),
                (float('inf'), 0.30)
            ],
            'income_tax_slabs_new': [
                (300000, 0.0),
                (700000, 0.05),
                (1000000, 0.10),
                (1200000, 0.15),
                (1500000, 0.20),
                (float('inf'), 0.30)
            ],
            'ltcg_equity': 0.10,  # Long term capital gains on equity
            'stcg_equity': 0.15,  # Short term capital gains on equity
            'ltcg_debt': 0.20,    # Long term capital gains on debt with indexation
            'tds_threshold': 40000,  # TDS threshold for interest
            'tds_threshold_senior': 50000,  # TDS threshold for senior citizens
            'section_80c_limit': 150000,
            'nps_additional_limit': 50000
        }
    
    def _load_current_interest_rates(self) -> Dict:
        """Load current interest rates for 2025"""
        return {
            'ppf': 7.1,
            'nsc': 6.8,
            'epf': 8.25,
            'fd_average': 7.0,
            'fd_small_finance': 8.5,
            'savings_account': 3.0,
            'inflation': 5.5
        }
    
    def calculate_compound_interest(self, principal: float, rate: float, 
                                 time_years: float, compounding_frequency: int = 1) -> float:
        """
        Calculate compound interest
        compounding_frequency: 1=annual, 2=semi-annual, 4=quarterly, 12=monthly
        """
        if rate <= 0 or time_years <= 0 or principal <= 0:
            return principal
        
        rate_decimal = rate / 100
        amount = principal * (1 + rate_decimal / compounding_frequency) ** (compounding_frequency * time_years)
        return round(amount, 2)
    
    def calculate_simple_interest(self, principal: float, rate: float, time_years: float) -> float:
        """Calculate simple interest"""
        if rate <= 0 or time_years <= 0 or principal <= 0:
            return principal
        
        interest = principal * (rate / 100) * time_years
        return round(principal + interest, 2)
    
    def calculate_fd_maturity(self, principal: float, rate: float, 
                            tenure_months: int, compounding: str = 'quarterly') -> float:
        """Calculate FD maturity amount with quarterly compounding (standard for most banks)"""
        if tenure_months <= 0 or rate <= 0 or principal <= 0:
            return principal
        
        time_years = tenure_months / 12
        
        if compounding == 'quarterly':
            return self.calculate_compound_interest(principal, rate, time_years, 4)
        elif compounding == 'monthly':
            return self.calculate_compound_interest(principal, rate, time_years, 12)
        else:  # annual
            return self.calculate_compound_interest(principal, rate, time_years, 1)
    
    def calculate_ppf_maturity(self, annual_investment: float, years: int = 15, 
                             rate: float = None) -> Dict:
        """Calculate PPF maturity with annual investments"""
        if rate is None:
            rate = self.current_interest_rates['ppf']
        
        # PPF allows annual investments, compounded annually
        total_investment = 0
        maturity_amount = 0
        
        for year in range(years):
            total_investment += annual_investment
            # Each year's investment compounds for remaining years
            remaining_years = years - year
            year_maturity = self.calculate_compound_interest(annual_investment, rate, remaining_years, 1)
            maturity_amount += year_maturity
        
        return {
            'total_investment': total_investment,
            'maturity_amount': round(maturity_amount, 2),
            'total_interest': round(maturity_amount - total_investment, 2),
            'effective_rate': round(((maturity_amount / total_investment) ** (1/years) - 1) * 100, 2)
        }
    
    def calculate_sip_maturity(self, monthly_investment: float, rate: float, 
                             years: int) -> Dict:
        """Calculate SIP maturity using future value of annuity formula"""
        if monthly_investment <= 0 or rate <= 0 or years <= 0:
            return {'maturity_amount': 0, 'total_investment': 0, 'total_returns': 0}
        
        monthly_rate = rate / (12 * 100)
        months = years * 12
        
        # Future value of ordinary annuity
        if monthly_rate > 0:
            maturity_amount = monthly_investment * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            maturity_amount = monthly_investment * months
        
        total_investment = monthly_investment * months
        total_returns = maturity_amount - total_investment
        
        return {
            'maturity_amount': round(maturity_amount, 2),
            'total_investment': round(total_investment, 2),
            'total_returns': round(total_returns, 2),
            'return_percentage': round((total_returns / total_investment) * 100, 2) if total_investment > 0 else 0
        }
    
    def calculate_loan_emi(self, principal: float, rate: float, tenure_years: int) -> Dict:
        """Calculate loan EMI using standard formula"""
        if principal <= 0 or rate <= 0 or tenure_years <= 0:
            return {'emi': 0, 'total_payment': 0, 'total_interest': 0}
        
        monthly_rate = rate / (12 * 100)
        months = tenure_years * 12
        
        if monthly_rate > 0:
            emi = principal * monthly_rate * (1 + monthly_rate) ** months / ((1 + monthly_rate) ** months - 1)
        else:
            emi = principal / months
        
        total_payment = emi * months
        total_interest = total_payment - principal
        
        return {
            'emi': round(emi, 2),
            'total_payment': round(total_payment, 2),
            'total_interest': round(total_interest, 2),
            'interest_percentage': round((total_interest / principal) * 100, 2)
        }
    
    def calculate_tax_liability(self, annual_income: float, regime: str = 'new', 
                              age: int = 30, deductions: Dict = None) -> Dict:
        """Calculate income tax liability"""
        if deductions is None:
            deductions = {}
        
        # Standard deduction
        standard_deduction = 75000 if regime == 'new' else 50000
        
        # Choose tax slabs
        if regime == 'new':
            tax_slabs = self.current_tax_rates['income_tax_slabs_new']
        else:
            tax_slabs = self.current_tax_rates['income_tax_slabs_old']
        
        # Calculate taxable income
        taxable_income = annual_income - standard_deduction
        
        if regime == 'old':
            # Apply deductions under old regime
            section_80c = min(deductions.get('80c', 0), self.current_tax_rates['section_80c_limit'])
            nps_additional = min(deductions.get('80ccd_1b', 0), self.current_tax_rates['nps_additional_limit'])
            other_deductions = deductions.get('other', 0)
            
            total_deductions = section_80c + nps_additional + other_deductions
            taxable_income = max(0, taxable_income - total_deductions)
        
        # Calculate tax
        tax = 0
        remaining_income = taxable_income
        
        for i, (slab_limit, rate) in enumerate(tax_slabs):
            if remaining_income <= 0:
                break
            
            if i == 0:
                taxable_in_slab = min(remaining_income, slab_limit)
            else:
                prev_limit = tax_slabs[i-1][0]
                taxable_in_slab = min(remaining_income, slab_limit - prev_limit)
            
            tax += taxable_in_slab * rate
            remaining_income -= taxable_in_slab
        
        # Add cess (4% on tax)
        cess = tax * 0.04
        total_tax = tax + cess
        
        # Rebate under section 87A (new regime)
        if regime == 'new' and annual_income <= 700000:
            rebate = min(total_tax, 25000)
            total_tax = max(0, total_tax - rebate)
        elif regime == 'old' and annual_income <= 500000:
            rebate = min(total_tax, 12500)
            total_tax = max(0, total_tax - rebate)
        
        return {
            'annual_income': annual_income,
            'taxable_income': taxable_income,
            'tax_before_cess': round(tax, 2),
            'cess': round(cess, 2),
            'total_tax': round(total_tax, 2),
            'effective_tax_rate': round((total_tax / annual_income) * 100, 2) if annual_income > 0 else 0,
            'tax_regime': regime
        }
    
    def calculate_capital_gains_tax(self, purchase_price: float, sale_price: float, 
                                  holding_period_days: int, asset_type: str) -> Dict:
        """Calculate capital gains tax"""
        capital_gain = sale_price - purchase_price
        
        if capital_gain <= 0:
            return {
                'capital_gain': capital_gain,
                'tax_type': 'Loss',
                'tax_amount': 0,
                'net_gain': capital_gain
            }
        
        is_long_term = False
        tax_rate = 0
        
        if asset_type.lower() in ['equity', 'stocks', 'mutual_funds_equity']:
            is_long_term = holding_period_days > 365
            if is_long_term:
                # LTCG on equity: 10% on gains above ₹1 lakh
                taxable_gain = max(0, capital_gain - 100000)
                tax_rate = self.current_tax_rates['ltcg_equity']
            else:
                # STCG on equity: 15%
                taxable_gain = capital_gain
                tax_rate = self.current_tax_rates['stcg_equity']
        
        elif asset_type.lower() in ['debt', 'mutual_funds_debt', 'bonds']:
            is_long_term = holding_period_days > (3 * 365)
            if is_long_term:
                # LTCG on debt: 20% with indexation
                tax_rate = self.current_tax_rates['ltcg_debt']
                # Simplified: assuming 5% annual indexation
                indexation_years = holding_period_days / 365
                indexed_cost = purchase_price * (1.05 ** indexation_years)
                taxable_gain = max(0, sale_price - indexed_cost)
            else:
                # STCG on debt: As per income tax slab
                taxable_gain = capital_gain
                tax_rate = 0.30  # Assuming highest slab
        
        elif asset_type.lower() in ['gold', 'silver', 'precious_metals']:
            is_long_term = holding_period_days > (3 * 365)
            if is_long_term:
                tax_rate = 0.20  # With indexation
                indexation_years = holding_period_days / 365
                indexed_cost = purchase_price * (1.05 ** indexation_years)
                taxable_gain = max(0, sale_price - indexed_cost)
            else:
                taxable_gain = capital_gain
                tax_rate = 0.30
        
        tax_amount = taxable_gain * tax_rate
        net_gain = capital_gain - tax_amount
        
        return {
            'capital_gain': round(capital_gain, 2),
            'taxable_gain': round(taxable_gain, 2),
            'tax_type': 'LTCG' if is_long_term else 'STCG',
            'tax_rate': tax_rate * 100,
            'tax_amount': round(tax_amount, 2),
            'net_gain': round(net_gain, 2),
            'holding_period_days': holding_period_days
        }
    
    def calculate_insurance_requirement(self, annual_income: float, age: int, 
                                      dependents: int, existing_assets: float = 0,
                                      existing_insurance: float = 0) -> Dict:
        """Calculate life insurance requirement using Human Life Value method"""
        
        # Working years remaining (assuming retirement at 60)
        working_years = max(0, 60 - age)
        
        # Annual income replacement (80% of current income)
        income_replacement = annual_income * 0.8
        
        # Present value of future income (assuming 6% discount rate, 5% income growth)
        discount_rate = 0.06
        growth_rate = 0.05
        net_rate = (discount_rate - growth_rate) / (1 + growth_rate)
        
        if net_rate > 0 and working_years > 0:
            pv_income = income_replacement * (1 - (1 + net_rate) ** (-working_years)) / net_rate
        else:
            pv_income = income_replacement * working_years
        
        # Additional coverage for dependents
        dependent_coverage = dependents * annual_income * 2
        
        # Education fund for children (assuming 2 children)
        education_fund = min(dependents, 2) * 1000000  # ₹10 lakh per child
        
        # Emergency fund
        emergency_fund = annual_income * 2
        
        # Total requirement
        total_requirement = pv_income + dependent_coverage + education_fund + emergency_fund
        
        # Subtract existing assets and insurance
        net_requirement = max(0, total_requirement - existing_assets - existing_insurance)
        
        return {
            'total_requirement': round(total_requirement, 0),
            'existing_coverage': existing_assets + existing_insurance,
            'additional_coverage_needed': round(net_requirement, 0),
            'recommended_term_insurance': round(net_requirement * 0.8, 0),
            'annual_premium_estimate': round(net_requirement * 0.001, 0),  # Rough estimate
            'components': {
                'income_replacement': round(pv_income, 0),
                'dependent_coverage': round(dependent_coverage, 0),
                'education_fund': round(education_fund, 0),
                'emergency_fund': round(emergency_fund, 0)
            }
        }
    
    def calculate_retirement_corpus(self, current_age: int, retirement_age: int = 60,
                                  current_expenses: float = 0, inflation_rate: float = None,
                                  life_expectancy: int = 80) -> Dict:
        """Calculate retirement corpus requirement"""
        
        if inflation_rate is None:
            inflation_rate = self.current_interest_rates['inflation']
        
        years_to_retirement = retirement_age - current_age
        retirement_years = life_expectancy - retirement_age
        
        if years_to_retirement <= 0:
            return {'message': 'Already at or past retirement age'}
        
        # If current expenses not provided, estimate as 70% of current income
        if current_expenses <= 0:
            current_expenses = 0  # Will need to be provided
        
        # Future expenses at retirement (adjusted for inflation)
        future_monthly_expenses = current_expenses * (1 + inflation_rate/100) ** years_to_retirement
        future_annual_expenses = future_monthly_expenses * 12
        
        # Corpus required (assuming 4% withdrawal rate in retirement)
        withdrawal_rate = 0.04
        required_corpus = future_annual_expenses / withdrawal_rate
        
        # Present value of required corpus
        discount_rate = 0.08  # Expected return rate
        pv_required_corpus = required_corpus / (1 + discount_rate) ** years_to_retirement
        
        return {
            'years_to_retirement': years_to_retirement,
            'current_monthly_expenses': current_expenses,
            'future_monthly_expenses': round(future_monthly_expenses, 0),
            'required_corpus': round(required_corpus, 0),
            'pv_required_corpus': round(pv_required_corpus, 0),
            'monthly_sip_required': round(self._calculate_sip_for_target(required_corpus, discount_rate * 100, years_to_retirement), 0)
        }
    
    def _calculate_sip_for_target(self, target_amount: float, rate: float, years: int) -> float:
        """Calculate monthly SIP required to reach target amount"""
        monthly_rate = rate / (12 * 100)
        months = years * 12
        
        if monthly_rate > 0:
            sip = target_amount * monthly_rate / ((1 + monthly_rate) ** months - 1)
        else:
            sip = target_amount / months
        
        return sip
    
    def calculate_emergency_fund(self, monthly_expenses: float, job_stability: str = 'stable',
                               dependents: int = 0) -> Dict:
        """Calculate emergency fund requirement"""
        
        base_months = 6  # Base requirement
        
        # Adjust based on job stability
        stability_multiplier = {
            'very_stable': 1.0,    # Government job
            'stable': 1.2,         # Private job, established business
            'moderate': 1.5,       # New business, gig economy
            'unstable': 2.0        # Daily wage, seasonal work
        }
        
        multiplier = stability_multiplier.get(job_stability.lower(), 1.2)
        
        # Adjust for dependents
        dependent_months = dependents * 1  # 1 additional month per dependent
        
        total_months = base_months * multiplier + dependent_months
        emergency_fund = monthly_expenses * total_months
        
        return {
            'recommended_months': round(total_months, 1),
            'emergency_fund_amount': round(emergency_fund, 0),
            'liquid_investment': round(emergency_fund * 0.7, 0),  # 70% in liquid funds
            'savings_account': round(emergency_fund * 0.3, 0),    # 30% in savings account
            'job_stability': job_stability,
            'base_calculation': f"{base_months} × {multiplier} + {dependent_months} dependent months"
        }
    
    def calculate_debt_consolidation(self, debts: List[Dict]) -> Dict:
        """Calculate debt consolidation benefits"""
        if not debts:
            return {'message': 'No debts provided'}
        
        total_debt = sum(debt['amount'] for debt in debts)
        total_emi = sum(debt['emi'] for debt in debts)
        weighted_rate = sum(debt['amount'] * debt['rate'] for debt in debts) / total_debt
        
        # Calculate consolidated loan at lower rate
        consolidated_rate = weighted_rate * 0.8  # Assume 20% rate reduction
        consolidated_tenure = max(debt.get('tenure_years', 5) for debt in debts)
        
        consolidated_emi_calc = self.calculate_loan_emi(total_debt, consolidated_rate, consolidated_tenure)
        consolidated_emi = consolidated_emi_calc['emi']
        
        monthly_savings = total_emi - consolidated_emi
        total_interest_old = sum(debt['emi'] * debt.get('tenure_years', 5) * 12 - debt['amount'] for debt in debts)
        total_interest_new = consolidated_emi_calc['total_interest']
        interest_savings = total_interest_old - total_interest_new
        
        return {
            'current_total_debt': round(total_debt, 0),
            'current_total_emi': round(total_emi, 0),
            'current_weighted_rate': round(weighted_rate, 2),
            'consolidated_rate': round(consolidated_rate, 2),
            'consolidated_emi': round(consolidated_emi, 0),
            'monthly_savings': round(monthly_savings, 0),
            'total_interest_savings': round(interest_savings, 0),
            'savings_percentage': round((monthly_savings / total_emi) * 100, 1) if total_emi > 0 else 0
        }
    
    def calculate_asset_allocation(self, age: int, risk_tolerance: str, 
                                 investment_horizon: int) -> Dict:
        """Calculate recommended asset allocation"""
        
        # Base equity allocation = 100 - age
        base_equity = max(0, min(80, 100 - age))
        
        # Adjust for risk tolerance
        risk_adjustments = {
            'low': -20,
            'medium': 0,
            'high': +15
        }
        
        equity_adjustment = risk_adjustments.get(risk_tolerance.lower(), 0)
        
        # Adjust for investment horizon
        if investment_horizon < 3:
            equity_adjustment -= 20
        elif investment_horizon > 10:
            equity_adjustment += 10
        
        equity_allocation = max(0, min(80, base_equity + equity_adjustment))
        debt_allocation = max(15, 70 - equity_allocation)
        gold_allocation = max(5, min(15, 100 - equity_allocation - debt_allocation))
        
        # Normalize to 100%
        total = equity_allocation + debt_allocation + gold_allocation
        if total != 100:
            factor = 100 / total
            equity_allocation = round(equity_allocation * factor)
            debt_allocation = round(debt_allocation * factor)
            gold_allocation = 100 - equity_allocation - debt_allocation
        
        return {
            'equity_percentage': equity_allocation,
            'debt_percentage': debt_allocation,
            'gold_percentage': gold_allocation,
            'rebalancing_frequency': 'Annual',
            'risk_level': risk_tolerance.title(),
            'time_horizon': f"{investment_horizon} years"
        }

if __name__ == '__main__':
    # Test the calculator
    calc = FinancialCalculator()
    
    # Test various calculations
    print("🧮 Financial Calculator Tests")
    print("=" * 50)
    
    # FD calculation
    fd_result = calc.calculate_fd_maturity(100000, 8.5, 12)
    print(f"FD Maturity (₹1L @ 8.5% for 1 year): ₹{fd_result:,.0f}")
    
    # SIP calculation
    sip_result = calc.calculate_sip_maturity(5000, 12, 10)
    print(f"SIP Maturity (₹5K/month @ 12% for 10 years): ₹{sip_result['maturity_amount']:,.0f}")
    
    # PPF calculation
    ppf_result = calc.calculate_ppf_maturity(150000, 15)
    print(f"PPF Maturity (₹1.5L/year for 15 years): ₹{ppf_result['maturity_amount']:,.0f}")
    
    # Tax calculation
    tax_result = calc.calculate_tax_liability(800000, 'new', 35)
    print(f"Tax on ₹8L income (new regime): ₹{tax_result['total_tax']:,.0f}")
    
    # Insurance requirement
    insurance_result = calc.calculate_insurance_requirement(600000, 35, 2)
    print(f"Life Insurance requirement: ₹{insurance_result['additional_coverage_needed']:,.0f}")
