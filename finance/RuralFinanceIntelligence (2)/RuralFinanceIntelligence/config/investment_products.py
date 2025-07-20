"""
Investment products configuration for rural India financial advisory system
Contains comprehensive product details, eligibility criteria, and features
"""

from typing import Dict, List, Any
from datetime import datetime

# Investment Products Configuration
INVESTMENT_PRODUCTS = {
    
    # EQUITY PRODUCTS
    "equity": {
        "large_cap_stocks": {
            "recommended_stocks": [
                {
                    "symbol": "RELIANCE.NS",
                    "company_name": "Reliance Industries Ltd",
                    "sector": "Oil & Gas",
                    "market_cap_category": "Large Cap",
                    "minimum_investment": 100,  # 1 share minimum
                    "recommended_for": ["Medium", "High"],
                    "risk_level": "Medium",
                    "liquidity": "High",
                    "dividend_yield": 0.3,
                    "pe_ratio_range": (20, 30),
                    "features": ["Blue chip stock", "Regular dividends", "High liquidity"]
                },
                {
                    "symbol": "TCS.NS",
                    "company_name": "Tata Consultancy Services",
                    "sector": "Information Technology",
                    "market_cap_category": "Large Cap",
                    "minimum_investment": 100,
                    "recommended_for": ["Medium", "High"],
                    "risk_level": "Medium",
                    "liquidity": "High",
                    "dividend_yield": 1.8,
                    "pe_ratio_range": (25, 35),
                    "features": ["IT sector leader", "Consistent performance", "Export earnings"]
                },
                {
                    "symbol": "HDFCBANK.NS",
                    "company_name": "HDFC Bank Ltd",
                    "sector": "Financial Services",
                    "market_cap_category": "Large Cap",
                    "minimum_investment": 100,
                    "recommended_for": ["Low", "Medium", "High"],
                    "risk_level": "Medium",
                    "liquidity": "High",
                    "dividend_yield": 1.2,
                    "pe_ratio_range": (18, 25),
                    "features": ["Banking leader", "Stable growth", "Rural presence"]
                },
                {
                    "symbol": "ICICIBANK.NS",
                    "company_name": "ICICI Bank Ltd",
                    "sector": "Financial Services",
                    "market_cap_category": "Large Cap",
                    "minimum_investment": 100,
                    "recommended_for": ["Low", "Medium", "High"],
                    "risk_level": "Medium",
                    "liquidity": "High",
                    "dividend_yield": 0.8,
                    "pe_ratio_range": (15, 22),
                    "features": ["Private banking", "Digital innovation", "Rural banking"]
                }
            ],
            "investment_guidelines": {
                "minimum_amount": 5000,
                "maximum_single_stock": 0.1,  # 10% of portfolio
                "diversification_minimum": 5,  # At least 5 stocks
                "rebalancing_frequency": "Quarterly"
            }
        },
        
        "mid_cap_stocks": {
            "characteristics": {
                "market_cap_range": (50000, 200000),  # Crores
                "expected_return": 15,
                "volatility": "High",
                "recommended_for": ["High"],
                "minimum_investment": 10000
            },
            "selection_criteria": [
                "Strong fundamentals",
                "Growing market share",
                "Experienced management",
                "Debt-to-equity < 0.5"
            ]
        }
    },
    
    # MUTUAL FUNDS
    "mutual_funds": {
        "equity_large_cap": [
            {
                "scheme_name": "SBI BlueChip Fund - Direct Plan - Growth",
                "amc": "SBI Mutual Fund",
                "category": "Large Cap",
                "minimum_sip": 500,
                "minimum_lumpsum": 5000,
                "expense_ratio": 0.8,
                "exit_load": "1% if redeemed within 1 year",
                "benchmark": "S&P BSE 100",
                "fund_manager": "R. Srinivasan",
                "aum": 25000,  # Crores
                "risk_level": "Medium",
                "recommended_for": ["Low", "Medium", "High"],
                "features": ["Large cap focus", "Consistent performance", "Low volatility"]
            },
            {
                "scheme_name": "ICICI Prudential Bluechip Fund - Direct Plan - Growth",
                "amc": "ICICI Prudential MF",
                "category": "Large Cap",
                "minimum_sip": 1000,
                "minimum_lumpsum": 5000,
                "expense_ratio": 1.0,
                "exit_load": "1% if redeemed within 1 year",
                "benchmark": "Nifty 100",
                "aum": 30000,
                "risk_level": "Medium",
                "recommended_for": ["Low", "Medium", "High"],
                "features": ["Diversified portfolio", "Professional management", "Regular dividends"]
            }
        ],
        
        "elss_funds": [
            {
                "scheme_name": "Axis Long Term Equity Fund - Direct Plan - Growth",
                "amc": "Axis Mutual Fund",
                "category": "ELSS",
                "minimum_sip": 500,
                "minimum_lumpsum": 500,
                "expense_ratio": 0.8,
                "lock_in_period": 3,  # years
                "tax_benefit": "80C deduction up to ₹1.5 lakh",
                "benchmark": "S&P BSE 200",
                "aum": 15000,
                "risk_level": "High",
                "recommended_for": ["Medium", "High"],
                "features": ["Tax saving", "Equity growth", "Shortest lock-in in 80C"]
            },
            {
                "scheme_name": "Mirae Asset Tax Saver Fund - Direct Plan - Growth",
                "amc": "Mirae Asset MF",
                "category": "ELSS",
                "minimum_sip": 1000,
                "minimum_lumpsum": 1000,
                "expense_ratio": 0.7,
                "lock_in_period": 3,
                "tax_benefit": "80C deduction up to ₹1.5 lakh",
                "aum": 12000,
                "risk_level": "High",
                "recommended_for": ["Medium", "High"],
                "features": ["Tax efficient", "Growth oriented", "Multi-cap approach"]
            }
        ],
        
        "debt_funds": [
            {
                "scheme_name": "SBI Magnum Gilt Fund - Direct Plan - Growth",
                "amc": "SBI Mutual Fund",
                "category": "Gilt Fund",
                "minimum_sip": 500,
                "minimum_lumpsum": 1000,
                "expense_ratio": 0.5,
                "risk_level": "Low",
                "duration": "Medium to Long",
                "credit_risk": "None (Government securities)",
                "recommended_for": ["Low", "Medium"],
                "features": ["Government securities", "Low credit risk", "Interest rate sensitive"]
            }
        ],
        
        "hybrid_funds": [
            {
                "scheme_name": "SBI Equity Hybrid Fund - Direct Plan - Growth",
                "amc": "SBI Mutual Fund",
                "category": "Aggressive Hybrid",
                "equity_allocation": "65-80%",
                "debt_allocation": "20-35%",
                "minimum_sip": 500,
                "minimum_lumpsum": 5000,
                "expense_ratio": 0.9,
                "risk_level": "Medium",
                "recommended_for": ["Low", "Medium"],
                "features": ["Balanced approach", "Lower volatility", "Debt cushion"]
            }
        ]
    },
    
    # FIXED DEPOSITS
    "fixed_deposits": {
        "small_finance_banks": [
            {
                "bank_name": "NorthEast Small Finance Bank",
                "bank_type": "Small Finance Bank",
                "rates": {
                    12: {"general": 8.5, "senior_citizen": 9.0},
                    18: {"general": 9.0, "senior_citizen": 9.5},
                    24: {"general": 8.8, "senior_citizen": 9.3},
                    36: {"general": 8.5, "senior_citizen": 9.0}
                },
                "minimum_amount": 1000,
                "maximum_amount": 500000000,  # No practical limit
                "features": [
                    "DICGC insured up to ₹5 lakh",
                    "Highest interest rates",
                    "Digital banking",
                    "Premature withdrawal allowed with penalty"
                ],
                "penalty": "1% on premature withdrawal",
                "auto_renewal": True,
                "loan_against_fd": 90  # % of FD amount
            },
            {
                "bank_name": "Unity Small Finance Bank",
                "bank_type": "Small Finance Bank",
                "rates": {
                    12: {"general": 8.25, "senior_citizen": 8.75},
                    24: {"general": 8.6, "senior_citizen": 9.1},
                    36: {"general": 8.6, "senior_citizen": 9.1}
                },
                "minimum_amount": 1000,
                "features": [
                    "DICGC insured",
                    "High returns",
                    "Mobile banking",
                    "Quick processing"
                ]
            }
        ],
        
        "private_banks": [
            {
                "bank_name": "ICICI Bank",
                "bank_type": "Private Bank",
                "rates": {
                    12: {"general": 6.6, "senior_citizen": 7.1},
                    24: {"general": 6.8, "senior_citizen": 7.3},
                    60: {"general": 6.6, "senior_citizen": 7.1}
                },
                "minimum_amount": 10000,
                "features": [
                    "Wide branch network",
                    "Premium banking services",
                    "Digital convenience",
                    "Multiple tenure options"
                ],
                "special_products": [
                    "Tax Saver FD (5 years lock-in)",
                    "Flexi FD (partial withdrawal)",
                    "Auto-sweep facility"
                ]
            },
            {
                "bank_name": "HDFC Bank",
                "bank_type": "Private Bank",
                "rates": {
                    12: {"general": 6.5, "senior_citizen": 7.0},
                    24: {"general": 6.75, "senior_citizen": 7.25}
                },
                "minimum_amount": 10000,
                "features": [
                    "Trusted brand",
                    "Excellent service",
                    "Online FD booking",
                    "Relationship benefits"
                ]
            }
        ],
        
        "public_banks": [
            {
                "bank_name": "State Bank of India",
                "bank_type": "Public Sector Bank",
                "rates": {
                    12: {"general": 6.8, "senior_citizen": 7.3},
                    24: {"general": 7.0, "senior_citizen": 7.5}
                },
                "minimum_amount": 1000,
                "features": [
                    "Government backing",
                    "Widest network",
                    "Rural presence",
                    "Trusted institution"
                ],
                "special_schemes": [
                    "SBI Wecare Deposit (for senior citizens)",
                    "SBI Amrit Kalash (special rates)",
                    "Tax Saver Scheme"
                ]
            }
        ]
    },
    
    # GOVERNMENT SCHEMES
    "government_schemes": {
        "post_office_schemes": [
            {
                "scheme_name": "National Savings Certificate (NSC)",
                "interest_rate": 6.8,
                "tenure_years": 5,
                "minimum_investment": 1000,
                "maximum_investment": None,
                "tax_benefits": [
                    "80C deduction on investment",
                    "Interest reinvestment qualifies for 80C"
                ],
                "features": [
                    "Government guarantee",
                    "Compound interest",
                    "No risk",
                    "Available at all post offices"
                ],
                "eligibility": "Indian residents above 10 years",
                "premature_closure": "Not allowed except in specific cases"
            },
            {
                "scheme_name": "Kisan Vikas Patra (KVP)",
                "interest_rate": 6.9,
                "doubling_period": 124,  # months
                "minimum_investment": 1000,
                "maximum_investment": None,
                "features": [
                    "Money doubles in ~10 years",
                    "Government backing",
                    "Transferable",
                    "No tax benefits"
                ]
            }
        ],
        
        "employee_schemes": [
            {
                "scheme_name": "Employees' Provident Fund (EPF)",
                "interest_rate": 8.25,
                "employee_contribution": 12,  # % of basic salary
                "employer_contribution": 12,
                "tax_benefits": [
                    "Employee contribution: 80C deduction",
                    "Interest: Tax-free",
                    "Withdrawal: Tax-free after 5 years"
                ],
                "eligibility": "Salaried employees in covered establishments",
                "features": [
                    "Guaranteed returns",
                    "Employer matching",
                    "Retirement corpus building"
                ]
            }
        ]
    },
    
    # GOLD INVESTMENTS
    "gold_investments": [
        {
            "product_type": "Gold ETF",
            "examples": ["HDFC Gold ETF", "SBI Gold ETF", "ICICI Prudential Gold ETF"],
            "minimum_investment": 1000,
            "demat_account_required": True,
            "expense_ratio": 0.5,
            "liquidity": "High (tradeable on stock exchange)",
            "storage_cost": "None",
            "making_charges": "None",
            "tax_treatment": "20% LTCG with indexation after 3 years",
            "features": [
                "Paper gold",
                "High purity (99.5%)",
                "No storage hassle",
                "Real-time trading"
            ]
        },
        {
            "product_type": "Digital Gold",
            "platforms": ["Paytm Gold", "Google Pay Gold", "PhonePe Gold"],
            "minimum_investment": 1,
            "storage_cost": "Free",
            "making_charges": "None",
            "buy_sell_spread": "2-3%",
            "features": [
                "Start with ₹1",
                "24K purity",
                "Instant buy/sell",
                "Digital storage"
            ]
        },
        {
            "product_type": "Gold Mutual Funds",
            "fund_of_funds": True,
            "underlying": "Gold ETFs",
            "minimum_sip": 500,
            "expense_ratio": 0.8,
            "demat_account_required": False,
            "features": [
                "No demat account needed",
                "SIP facility",
                "Professional management"
            ]
        }
    ],
    
    # INSURANCE PRODUCTS
    "insurance": {
        "term_life_insurance": {
            "recommended_providers": [
                {
                    "provider": "LIC",
                    "product_name": "LIC Tech Term",
                    "features": ["Online process", "Lower premiums", "Claim settlement ratio: 98%"],
                    "premium_estimate": "₹500-2000 per lakh coverage"
                },
                {
                    "provider": "HDFC Life",
                    "product_name": "Click 2 Protect Life",
                    "features": ["Return of premium option", "Waiver of premium", "Online discounts"],
                    "premium_estimate": "₹600-2500 per lakh coverage"
                }
            ],
            "coverage_calculation": "10-15 times annual income",
            "recommended_tenure": "Till retirement age"
        },
        
        "government_insurance": [
            {
                "scheme_name": "Pradhan Mantri Jeevan Jyoti Bima Yojana (PMJJBY)",
                "coverage": 200000,
                "premium": 330,
                "age_limit": "18-50 years",
                "eligibility": "Bank account holder",
                "auto_debit": True,
                "tax_benefit": "80C deduction",
                "features": ["Low cost", "Easy enrollment", "Death coverage"]
            },
            {
                "scheme_name": "Pradhan Mantri Suraksha Bima Yojana (PMSBY)",
                "coverage": 200000,
                "premium": 12,
                "age_limit": "18-70 years",
                "coverage_type": "Accidental death & disability",
                "features": ["Very low premium", "Easy enrollment", "Accident coverage"]
            }
        ]
    }
}

# Product selection criteria based on user profile
PRODUCT_SELECTION_CRITERIA = {
    "risk_based": {
        "Low": {
            "equity_allocation": "0-20%",
            "preferred_products": [
                "Fixed Deposits",
                "Government Schemes",
                "Debt Mutual Funds",
                "PPF"
            ],
            "avoid_products": ["Mid Cap Stocks", "Small Cap Funds"]
        },
        "Medium": {
            "equity_allocation": "30-60%",
            "preferred_products": [
                "Large Cap Mutual Funds",
                "ELSS Funds",
                "Hybrid Funds",
                "Blue Chip Stocks"
            ]
        },
        "High": {
            "equity_allocation": "60-80%",
            "preferred_products": [
                "Equity Mutual Funds",
                "Individual Stocks",
                "ELSS Funds",
                "Mid Cap Funds"
            ]
        }
    },
    
    "income_based": {
        "below_25000": {
            "focus": "Emergency fund and basic insurance",
            "recommended_start": ["SIP in ELSS", "PMJJBY", "PMSBY"],
            "minimum_amounts": 500
        },
        "25000_to_50000": {
            "focus": "Systematic investing and insurance",
            "recommended_products": ["SIP", "Term Insurance", "PPF"],
            "insurance_coverage": "5-8 times annual income"
        },
        "above_50000": {
            "focus": "Wealth creation and tax planning",
            "recommended_products": ["Diversified portfolio", "Direct stocks", "Real estate"],
            "insurance_coverage": "10-15 times annual income"
        }
    },
    
    "age_based": {
        "below_30": {
            "equity_allocation": "70-80%",
            "focus": "Growth and wealth creation",
            "time_horizon": "Long term"
        },
        "30_to_45": {
            "equity_allocation": "50-70%",
            "focus": "Balanced growth with some stability",
            "considerations": ["Children education", "Home loan"]
        },
        "45_to_60": {
            "equity_allocation": "30-50%",
            "focus": "Retirement planning with reduced risk",
            "priority": ["Debt reduction", "Retirement corpus"]
        },
        "above_60": {
            "equity_allocation": "10-30%",
            "focus": "Capital preservation and income generation",
            "priority": ["Fixed income", "Healthcare", "Liquidity"]
        }
    }
}

# Investment product features and constraints
PRODUCT_CONSTRAINTS = {
    "minimum_investments": {
        "mutual_fund_sip": 500,
        "mutual_fund_lumpsum": 5000,
        "direct_equity": 5000,
        "fixed_deposit": 1000,
        "ppf": 500,
        "nsc": 1000,
        "gold_etf": 1000,
        "digital_gold": 1
    },
    
    "lock_in_periods": {
        "elss": 3,  # years
        "ppf": 15,
        "nsc": 5,
        "epf": "Till retirement",
        "nps": "Till 60 years"
    },
    
    "liquidity_ratings": {
        "savings_account": 10,
        "liquid_funds": 9,
        "equity_stocks": 8,
        "equity_mutual_funds": 8,
        "gold_etf": 8,
        "fixed_deposits": 5,
        "debt_mutual_funds": 7,
        "ppf": 3,
        "nsc": 2,
        "elss": 2
    }
}

if __name__ == '__main__':
    # Print product summary
    print("📊 Investment Products Summary")
    print("=" * 50)
    
    equity_products = len(INVESTMENT_PRODUCTS['equity']['large_cap_stocks']['recommended_stocks'])
    mf_products = len(INVESTMENT_PRODUCTS['mutual_funds']['equity_large_cap'])
    fd_banks = len(INVESTMENT_PRODUCTS['fixed_deposits']['small_finance_banks'])
    
    print(f"Equity Products: {equity_products}")
    print(f"Mutual Fund Products: {mf_products}")
    print(f"FD Options: {fd_banks}")
    print(f"Government Schemes: {len(INVESTMENT_PRODUCTS['government_schemes']['post_office_schemes'])}")
    print(f"Gold Investment Options: {len(INVESTMENT_PRODUCTS['gold_investments'])}")
