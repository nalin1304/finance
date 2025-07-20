"""
Government schemes configuration for rural India financial advisory system
Contains comprehensive information about central and state government schemes
"""

from typing import Dict, List, Any
from datetime import datetime

# Government Schemes Configuration
GOVERNMENT_SCHEMES = {
    
    # CENTRAL GOVERNMENT SCHEMES - AGRICULTURE & RURAL
    "agriculture_rural": {
        "pm_kisan": {
            "full_name": "Pradhan Mantri Kisan Samman Nidhi",
            "launched_year": 2019,
            "ministry": "Ministry of Agriculture and Farmers Welfare",
            "scheme_type": "Direct Benefit Transfer",
            "financial_benefit": {
                "amount": 6000,
                "frequency": "Annual",
                "installments": 3,
                "installment_amount": 2000,
                "payment_months": ["April", "August", "December"]
            },
            "eligibility_criteria": {
                "beneficiary_type": "Small and marginal farmers",
                "land_holding": "Up to 2 hectares",
                "family_definition": "Husband, wife and minor children",
                "exclusions": [
                    "Income tax payers",
                    "Government employees",
                    "Retired pensioners with >₹10,000/month",
                    "Constitutional post holders",
                    "Professional tax payers"
                ]
            },
            "required_documents": [
                "Aadhaar card",
                "Bank account details",
                "Land ownership documents",
                "Citizenship certificate"
            ],
            "application_process": {
                "online": "pmkisan.gov.in",
                "offline": "Village Revenue Officer/Patwari",
                "csc_centers": True,
                "documents_verification": "Village level"
            },
            "implementation_agency": "State Governments",
            "current_beneficiaries": 11000000,  # 11 crores as of 2025
            "budget_allocation_2025": 68000000000,  # ₹68,000 crores
            "status": "Active",
            "features": [
                "Direct bank transfer",
                "No intermediary",
                "Aadhaar-linked",
                "Self-registration available"
            ]
        },
        
        "pmfby": {
            "full_name": "Pradhan Mantri Fasal Bima Yojana",
            "launched_year": 2016,
            "scheme_type": "Crop Insurance",
            "coverage": {
                "crops_covered": ["Kharif", "Rabi", "Commercial/Horticultural"],
                "risk_coverage": [
                    "Drought", "Flood", "Cyclone", "Pest attack",
                    "Disease", "Landslide", "Natural fire"
                ],
                "sum_insured": "Based on scale of finance for crop"
            },
            "premium_rates": {
                "kharif_crops": 2.0,  # % of sum insured
                "rabi_crops": 1.5,
                "commercial_horticultural": 5.0
            },
            "government_subsidy": {
                "central_share": 50,  # %
                "state_share": 50,
                "farmer_contribution": "As per premium rates above"
            },
            "eligibility": {
                "all_farmers": True,
                "sharecroppers": True,
                "tenant_farmers": True,
                "compulsory_for": "Loanee farmers"
            },
            "claim_settlement": {
                "basis": "Area approach",
                "technology": "Satellite imagery, weather stations",
                "settlement_period": "Within 2 months"
            },
            "exclusions": [
                "War and nuclear risks",
                "Malicious damage",
                "Theft or acts of mischief"
            ]
        },
        
        "soil_health_card": {
            "full_name": "Soil Health Card Scheme",
            "launched_year": 2015,
            "objective": "Promote soil test based nutrient management",
            "financial_benefit": {
                "testing_cost": "Free for farmers",
                "frequency": "Every 3 years",
                "subsidy_on_inputs": "Up to 50% on recommended fertilizers"
            },
            "services_provided": [
                "Soil testing",
                "Nutrient status analysis",
                "Fertilizer recommendations",
                "Organic matter content",
                "pH level assessment"
            ],
            "implementation": "State agriculture departments",
            "target": "14 crores farm holdings"
        }
    },
    
    # SOCIAL SECURITY SCHEMES
    "social_security": {
        "pmjjby": {
            "full_name": "Pradhan Mantri Jeevan Jyoti Bima Yojana",
            "launched_year": 2015,
            "scheme_type": "Life Insurance",
            "coverage_amount": 200000,
            "premium": {
                "annual_premium": 330,
                "deduction_date": "31st May annually",
                "auto_debit": True
            },
            "age_criteria": {
                "minimum_age": 18,
                "maximum_age": 50,
                "coverage_till": 55
            },
            "eligibility": {
                "bank_account": "Required",
                "aadhaar": "Required",
                "consent": "Auto-debit consent required"
            },
            "coverage_details": {
                "death_benefit": 200000,
                "cause": "Any reason",
                "beneficiary": "Nominee"
            },
            "enrollment": {
                "period": "1st June to 31st May",
                "channels": ["Bank branches", "CSC", "Online"],
                "documents": ["Bank account", "Aadhaar", "Consent form"]
            },
            "claim_process": {
                "intimation_period": "Within 30 days",
                "documents_required": [
                    "Death certificate", "Claim form",
                    "Cancelled cheque", "Nominee ID proof"
                ],
                "settlement_time": "30 days from claim submission"
            },
            "tax_benefits": "Premium eligible for 80C deduction"
        },
        
        "pmsby": {
            "full_name": "Pradhan Mantri Suraksha Bima Yojana",
            "launched_year": 2015,
            "scheme_type": "Accident Insurance",
            "coverage_amount": {
                "accidental_death": 200000,
                "permanent_total_disability": 200000,
                "permanent_partial_disability": 100000
            },
            "premium": {
                "annual_premium": 12,
                "deduction_date": "1st June annually",
                "auto_debit": True
            },
            "age_criteria": {
                "minimum_age": 18,
                "maximum_age": 70
            },
            "coverage_conditions": {
                "accident_definition": "Sudden, unforeseen, involuntary event",
                "death_within": "180 days of accident",
                "disability_certification": "Civil surgeon certificate required"
            },
            "exclusions": [
                "Suicide or self-inflicted injury",
                "War or war-like activities",
                "Nuclear risks",
                "Intoxication",
                "Adventure sports"
            ]
        },
        
        "apy": {
            "full_name": "Atal Pension Yojana",
            "launched_year": 2015,
            "scheme_type": "Pension Scheme",
            "target_group": "Unorganized sector workers",
            "age_criteria": {
                "joining_age": "18-40 years",
                "pension_age": 60
            },
            "pension_options": {
                "monthly_pension": [1000, 2000, 3000, 4000, 5000],
                "contribution_varies": "Based on joining age and pension amount"
            },
            "government_co_contribution": {
                "period": "5 years (2015-2020)",
                "condition": "50% of contribution or ₹1000, whichever is lower",
                "eligibility": "Not income tax payer"
            },
            "death_benefits": {
                "during_contribution": "Spouse continues or withdraws corpus",
                "during_pension": "Spouse gets same pension",
                "both_dead": "Nominee gets accumulated corpus"
            },
            "withdrawal": {
                "before_60": "Only in exceptional circumstances",
                "penalty": "Applicable on premature exit",
                "after_60": "Monthly pension guaranteed"
            }
        }
    },
    
    # FINANCIAL INCLUSION SCHEMES
    "financial_inclusion": {
        "jan_dhan": {
            "full_name": "Pradhan Mantri Jan Dhan Yojana",
            "launched_year": 2014,
            "objective": "Financial inclusion of all households",
            "account_features": {
                "zero_balance": True,
                "overdraft_facility": 10000,
                "overdraft_after": "6 months of satisfactory operation",
                "aadhaar_seeding": "Mandatory",
                "mobile_banking": True
            },
            "insurance_benefits": {
                "life_insurance": 30000,
                "accident_insurance": 100000,
                "premium": "Government funded"
            },
            "debit_card": {
                "rupay_card": "Free",
                "accident_insurance": 100000,
                "life_insurance": 30000
            },
            "direct_benefit_transfer": {
                "subsidy_transfer": "Direct to account",
                "scholarship": "Direct transfer",
                "pension": "Direct transfer"
            },
            "achievements_2025": {
                "accounts_opened": 460000000,  # 46 crore
                "deposits": 1900000000000,  # ₹1.9 lakh crore
                "rural_accounts": "56% of total"
            }
        },
        
        "mudra": {
            "full_name": "Pradhan Mantri Mudra Yojana",
            "launched_year": 2015,
            "objective": "Funding micro enterprises",
            "loan_categories": {
                "shishu": {
                    "amount_range": [1, 50000],
                    "target": "Starting stage businesses"
                },
                "kishore": {
                    "amount_range": [50001, 500000],
                    "target": "Growing businesses"
                },
                "tarun": {
                    "amount_range": [500001, 1000000],
                    "target": "Established businesses"
                }
            },
            "interest_rates": {
                "range": "Bank's base rate + spread",
                "typical_range": [8, 12],  # %
                "no_processing_fee": "For loans up to ₹50,000"
            },
            "eligibility": {
                "individuals": True,
                "proprietorship": True,
                "partnership": True,
                "llp": True,
                "pvt_ltd": True,
                "manufacturing": True,
                "trading": True,
                "services": True
            },
            "collateral": {
                "upto_10_lakh": "No collateral required",
                "guarantee": "Credit guarantee fund support"
            },
            "lending_institutions": [
                "Commercial banks", "RRBs", "Small finance banks",
                "MFIs", "NBFCs"
            ]
        }
    },
    
    # EMPLOYMENT SCHEMES
    "employment": {
        "mgnrega": {
            "full_name": "Mahatma Gandhi National Rural Employment Guarantee Act",
            "launched_year": 2005,
            "guarantee": "100 days of wage employment per household",
            "wage_rates_2025": {
                "andhra_pradesh": 281,
                "bihar": 220,
                "gujarat": 258,
                "haryana": 357,
                "karnataka": 319,
                "kerala": 349,
                "madhya_pradesh": 221,
                "maharashtra": 285,
                "punjab": 309,
                "rajasthan": 255,
                "tamil_nadu": 305,
                "uttar_pradesh": 230,
                "west_bengal": 231
            },
            "eligibility": {
                "rural_households": True,
                "adult_members": "Willing to do unskilled manual work",
                "job_card": "Required",
                "minimum_age": 18
            },
            "work_types": [
                "Water conservation", "Drought proofing",
                "Rural connectivity", "Land development",
                "Flood control", "Rural sanitation"
            ],
            "payment": {
                "method": "Direct bank transfer",
                "frequency": "Weekly",
                "delay_compensation": "0.05% per day after 15 days"
            },
            "women_participation": "At least 33% mandate",
            "budget_2025": 730000000000  # ₹73,000 crores
        },
        
        "pmegp": {
            "full_name": "Prime Minister's Employment Generation Programme",
            "implementing_agency": "KVIC",
            "objective": "Generate employment through micro enterprises",
            "loan_amount": {
                "manufacturing": [100000, 2500000],
                "service": [100000, 1000000]
            },
            "subsidy_rates": {
                "general_category": {
                    "urban": 15,  # % of project cost
                    "rural": 25
                },
                "special_category": {
                    "urban": 25,
                    "rural": 35
                }
            },
            "beneficiary_contribution": {
                "general": [5, 10],  # % min
                "special": [5, 10]
            },
            "special_categories": [
                "SC/ST", "OBC", "Minorities", "Women",
                "Ex-servicemen", "Physically handicapped",
                "NER", "Hill and border areas"
            ],
            "loan_tenure": "3-7 years",
            "moratorium_period": "6-18 months"
        }
    },
    
    # EDUCATION SCHEMES
    "education": {
        "scholarship_schemes": [
            {
                "scheme_name": "Post Matric Scholarship for SC students",
                "target": "SC students for higher education",
                "financial_support": {
                    "maintenance_allowance": "₹380-1200 per month",
                    "compulsory_fees": "Actual fees",
                    "book_allowance": "₹750-1000"
                },
                "eligibility": {
                    "caste": "Scheduled Caste",
                    "income_limit": 250000,  # Family income
                    "education_level": "Post matriculation"
                }
            },
            {
                "scheme_name": "Post Matric Scholarship for OBC students",
                "target": "OBC students for higher education",
                "financial_support": {
                    "maintenance_allowance": "₹380-1200 per month",
                    "compulsory_fees": "Actual fees"
                },
                "eligibility": {
                    "caste": "Other Backward Classes",
                    "income_limit": 100000,
                    "education_level": "Post matriculation"
                }
            }
        ],
        
        "samagra_shiksha": {
            "full_name": "Samagra Shiksha Abhiyan",
            "scope": "School education from pre-school to class 12",
            "components": [
                "Sarva Shiksha Abhiyan (Elementary)",
                "Rashtriya Madhyamik Shiksha Abhiyan (Secondary)",
                "Teacher Education"
            ],
            "interventions": [
                "Infrastructure development",
                "Teacher training",
                "Digital education",
                "Inclusive education",
                "Vocational education"
            ]
        }
    },
    
    # HEALTH SCHEMES
    "health": {
        "ayushman_bharat": {
            "full_name": "Pradhan Mantri Jan Arogya Yojana",
            "launched_year": 2018,
            "coverage": {
                "amount": 500000,  # Per family per year
                "families_covered": 107000000,  # 10.7 crore families
                "beneficiaries": 500000000  # 50 crore individuals
            },
            "eligibility": {
                "basis": "SECC 2011 database",
                "rural_criteria": "Deprivation criteria",
                "urban_criteria": "Occupational criteria",
                "card_required": "Ayushman Card"
            },
            "services_covered": [
                "Hospitalization", "Pre-hospitalization",
                "Post-hospitalization", "Day care procedures",
                "Emergency services"
            ],
            "network": {
                "hospitals": 24000,  # Empanelled hospitals
                "public_private": "Both included"
            },
            "features": [
                "Cashless treatment",
                "Portable across India",
                "No premium payment",
                "Pre-existing conditions covered"
            ]
        }
    },
    
    # HOUSING SCHEMES
    "housing": {
        "pmay_rural": {
            "full_name": "Pradhan Mantri Awaas Yojana - Gramin",
            "launched_year": 2016,
            "target": "Housing for All by 2022",
            "financial_assistance": {
                "plain_areas": 120000,
                "hilly_states": 130000,
                "iap_areas": 130000  # Integrated Action Plan areas
            },
            "cost_sharing": {
                "central_share": 60,  # %
                "state_share": 40
            },
            "beneficiary_selection": {
                "basis": "SECC 2011",
                "verification": "Gram Sabha",
                "priority": [
                    "Homeless", "Zero or single room",
                    "Households with disabled member",
                    "SC/ST households"
                ]
            },
            "house_specifications": {
                "minimum_area": "25 sq. meters",
                "kitchen": "Dedicated cooking area",
                "toilet": "Individual toilet"
            },
            "convergence": [
                "MGNREGA (90 days wage)",
                "Swachh Bharat Mission (toilet)",
                "Deen Dayal Upadhyaya Grameen Kaushalya Yojana"
            ]
        }
    },
    
    # WOMEN EMPOWERMENT
    "women_empowerment": {
        "beti_bachao_beti_padhao": {
            "full_name": "Beti Bachao Beti Padhao",
            "launched_year": 2015,
            "objective": "Address declining child sex ratio",
            "components": [
                "Prevention of gender biased sex selection",
                "Ensuring survival and protection of girl child",
                "Ensuring education and participation of girl child"
            ],
            "implementation": "Multi-sectoral approach",
            "focus_districts": 640
        },
        
        "sukanya_samriddhi": {
            "full_name": "Sukanya Samriddhi Yojana",
            "launched_year": 2015,
            "target": "Girl child (below 10 years)",
            "account_features": {
                "minimum_deposit": 250,
                "maximum_deposit": 150000,  # Per year
                "maturity_period": 21,  # Years from account opening
                "interest_rate_2025": 8.0,  # % per annum
                "compounding": "Annual"
            },
            "tax_benefits": [
                "Deposit: 80C deduction",
                "Interest: Tax-free",
                "Maturity: Tax-free"
            ],
            "partial_withdrawal": {
                "allowed_after": 18,  # Years
                "amount": "50% of balance",
                "purpose": "Higher education or marriage"
            },
            "premature_closure": {
                "after_5_years": "Allowed in exceptional cases",
                "interest_rate": "Post office savings rate"
            }
        }
    }
}

# Scheme eligibility matrix based on user profiles
ELIGIBILITY_MATRIX = {
    "income_based": {
        "below_50000": [
            "PM-Kisan", "MGNREGA", "Jan Dhan", "PMJJBY", "PMSBY",
            "Ayushman Bharat", "PMAY-Rural", "Mudra-Shishu"
        ],
        "50000_to_200000": [
            "PM-Kisan", "PMFBY", "PMJJBY", "PMSBY", "APY",
            "Mudra-Kishore", "PMEGP"
        ],
        "200000_to_500000": [
            "PMFBY", "PMSBY", "APY", "Mudra-Tarun", "PMEGP"
        ],
        "above_500000": [
            "PMFBY", "PMSBY", "Mudra-Tarun"
        ]
    },
    
    "occupation_based": {
        "farmer": [
            "PM-Kisan", "PMFBY", "Soil Health Card",
            "KCC", "PMSBY", "APY"
        ],
        "laborer": [
            "MGNREGA", "PMJJBY", "PMSBY", "APY",
            "Jan Dhan", "Ayushman Bharat"
        ],
        "small_business": [
            "Mudra", "PMEGP", "PMSBY", "APY"
        ],
        "unemployed": [
            "MGNREGA", "PMEGP", "Skill Development"
        ]
    },
    
    "demographic_based": {
        "women": [
            "Beti Bachao Beti Padhao", "Sukanya Samriddhi",
            "SHG schemes", "Women-specific Mudra"
        ],
        "sc_st": [
            "Special scholarships", "SCA to SCSP",
            "PMAY priority", "Reserved quotas"
        ],
        "senior_citizens": [
            "Pension schemes", "PMSBY (till 70)",
            "Senior citizen savings schemes"
        ]
    }
}

# Application process templates
APPLICATION_PROCESSES = {
    "online_portals": {
        "pm_kisan": {
            "website": "pmkisan.gov.in",
            "mobile_app": "PM Kisan Mobile App",
            "documents_upload": True,
            "status_tracking": True
        },
        "ayushman_bharat": {
            "website": "pmjay.gov.in",
            "beneficiary_identification": "Name and mobile number",
            "hospital_finder": True
        },
        "mudra": {
            "website": "mudra.org.in",
            "online_application": True,
            "bank_connect": True
        }
    },
    
    "offline_channels": {
        "csc_centers": [
            "PM-Kisan registration",
            "Jan Dhan account opening",
            "Insurance enrollment",
            "Certificate downloads"
        ],
        "bank_branches": [
            "Jan Dhan accounts",
            "Insurance schemes",
            "Mudra loans",
            "APY enrollment"
        ],
        "gram_panchayat": [
            "MGNREGA job cards",
            "PMAY applications",
            "BPL certificates",
            "Caste certificates"
        ]
    }
}

# Benefits calculation formulas
BENEFIT_CALCULATORS = {
    "pm_kisan_annual": lambda land_hectares: 6000 if land_hectares <= 2 else 0,
    "mgnrega_annual": lambda days_worked, wage_rate: min(days_worked, 100) * wage_rate,
    "mudra_loan_eligibility": lambda business_type, annual_income: {
        "shishu": 50000 if annual_income < 100000 else 0,
        "kishore": 500000 if 100000 <= annual_income < 500000 else 0,
        "tarun": 1000000 if annual_income >= 500000 else 0
    }.get(business_type, 0),
    "sukanya_samriddhi_maturity": lambda annual_deposit, years: (
        annual_deposit * (((1.08 ** years) - 1) / 0.08) * 1.08
    )
}

if __name__ == '__main__':
    # Print schemes summary
    print("🏛️ Government Schemes Summary")
    print("=" * 50)
    
    total_schemes = 0
    for category, schemes in GOVERNMENT_SCHEMES.items():
        category_count = len(schemes)
        total_schemes += category_count
        print(f"{category.replace('_', ' ').title()}: {category_count} schemes")
    
    print(f"\nTotal Schemes: {total_schemes}")
    print(f"Eligibility Categories: {len(ELIGIBILITY_MATRIX)}")
    print(f"Application Channels: {len(APPLICATION_PROCESSES)}")
    
    # Example eligibility check
    print("\nExample: Farmer with 1.5 hectares, income ₹80,000")
    eligible_schemes = ELIGIBILITY_MATRIX["income_based"]["below_50000"] + \
                     ELIGIBILITY_MATRIX["occupation_based"]["farmer"]
    print(f"Eligible for: {', '.join(set(eligible_schemes))}")
