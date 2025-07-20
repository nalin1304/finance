
#!/usr/bin/env python3
"""
Simple Command-Line Financial Advisor
Demonstrates core functionality without Streamlit dependency
"""

import sys
import os
sys.path.append('RuralFinanceIntelligence (2)/RuralFinanceIntelligence')

try:
    from enhanced_data_generator import EnhancedRuralDataGenerator
    from market_data_fetcher import MarketDataFetcher
    from recommendation_engine import AdvancedRecommendationEngine
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Creating minimal demo...")

def create_sample_profile():
    """Create a sample user profile for demonstration"""
    return {
        'Age': 35,
        'Gender': 'Male',
        'Marital_Status': 'Married',
        'Dependents': 2,
        'Education_Level': 'Secondary',
        'State': 'Bihar',
        'Occupation': 'Marginal Farmer',
        'Monthly_Income_Primary': 25000,
        'Monthly_Income_Secondary': 5000,
        'Land_Owned_Acres': 2.0,
        'Total_Debt_Amount': 100000,
        'Gold_Grams': 50,
        'Bank_Distance_KM': 5.0,
        'Risk_Tolerance': 'Medium',
        'Investment_Horizon': 'Medium Term (2-5 years)',
        'Financial_Goal': 'Emergency Fund',
        'Monthly_Investment_Capacity': 5000,
        'UPI_Usage': True,
        'Smartphone_Access': True,
        'Internet_Access': True,
        'Bank_Account_Type': 'Regular Savings'
    }

def simple_recommendation_logic(profile, investment_amount=50000):
    """Simple rule-based recommendation logic"""
    recommendations = {
        'emergency_fund': 0,
        'fixed_deposits': 0,
        'government_schemes': 0,
        'mutual_funds': 0,
        'gold': 0
    }
    
    # Basic allocation based on risk tolerance and profile
    if profile['Risk_Tolerance'] == 'Low':
        recommendations['emergency_fund'] = investment_amount * 0.4
        recommendations['fixed_deposits'] = investment_amount * 0.3
        recommendations['government_schemes'] = investment_amount * 0.2
        recommendations['gold'] = investment_amount * 0.1
    elif profile['Risk_Tolerance'] == 'Medium':
        recommendations['emergency_fund'] = investment_amount * 0.3
        recommendations['fixed_deposits'] = investment_amount * 0.25
        recommendations['government_schemes'] = investment_amount * 0.25
        recommendations['mutual_funds'] = investment_amount * 0.15
        recommendations['gold'] = investment_amount * 0.05
    else:  # High risk
        recommendations['emergency_fund'] = investment_amount * 0.2
        recommendations['fixed_deposits'] = investment_amount * 0.2
        recommendations['mutual_funds'] = investment_amount * 0.4
        recommendations['government_schemes'] = investment_amount * 0.15
        recommendations['gold'] = investment_amount * 0.05
    
    return recommendations

def print_recommendations(profile, recommendations, investment_amount):
    """Print formatted recommendations"""
    print("\n" + "="*60)
    print("🏦 AI FINANCIAL ADVISOR FOR RURAL INDIA")
    print("="*60)
    
    # User Profile Summary
    print(f"\n👤 USER PROFILE:")
    print(f"   Name: {profile['Occupation']} from {profile['State']}")
    print(f"   Age: {profile['Age']}, {profile['Marital_Status']}")
    print(f"   Monthly Income: ₹{profile['Monthly_Income_Primary'] + profile['Monthly_Income_Secondary']:,}")
    print(f"   Risk Tolerance: {profile['Risk_Tolerance']}")
    print(f"   Investment Goal: {profile['Financial_Goal']}")
    
    # Investment Recommendations
    print(f"\n💰 INVESTMENT RECOMMENDATIONS (₹{investment_amount:,}):")
    print("-" * 50)
    
    total_allocated = 0
    for category, amount in recommendations.items():
        if amount > 0:
            percentage = (amount / investment_amount) * 100
            print(f"   {category.replace('_', ' ').title():<20}: ₹{amount:>8,.0f} ({percentage:>5.1f}%)")
            total_allocated += amount
    
    print("-" * 50)
    print(f"   {'Total Allocated':<20}: ₹{total_allocated:>8,.0f} ({100.0:>5.1f}%)")
    
    # Specific Product Recommendations
    print(f"\n📋 SPECIFIC PRODUCTS:")
    print("-" * 50)
    
    if recommendations['emergency_fund'] > 0:
        print(f"   • High-yield Savings Account: ₹{recommendations['emergency_fund']:,.0f}")
        print(f"     - Keep in easily accessible savings account")
        print(f"     - Target: 6 months of expenses")
    
    if recommendations['fixed_deposits'] > 0:
        print(f"   • Fixed Deposits: ₹{recommendations['fixed_deposits']:,.0f}")
        print(f"     - Split across 2-3 banks for better rates")
        print(f"     - Consider laddering with different maturities")
    
    if recommendations['government_schemes'] > 0:
        print(f"   • Government Schemes: ₹{recommendations['government_schemes']:,.0f}")
        print(f"     - PPF: ₹{min(150000, recommendations['government_schemes'] * 0.6):,.0f} (15-year lock-in)")
        print(f"     - NSC: ₹{recommendations['government_schemes'] * 0.4:,.0f} (5-year lock-in)")
    
    if recommendations['mutual_funds'] > 0:
        print(f"   • Mutual Funds: ₹{recommendations['mutual_funds']:,.0f}")
        print(f"     - ELSS Funds: ₹{recommendations['mutual_funds'] * 0.6:,.0f} (Tax saving)")
        print(f"     - Balanced Funds: ₹{recommendations['mutual_funds'] * 0.4:,.0f}")
    
    if recommendations['gold'] > 0:
        print(f"   • Gold Investment: ₹{recommendations['gold']:,.0f}")
        print(f"     - Gold ETF or Digital Gold preferred over physical")
    
    # Next Steps
    print(f"\n✅ NEXT STEPS:")
    print("-" * 50)
    print("   1. Open a bank account if you don't have one")
    print("   2. Get your KYC documents ready (Aadhaar, PAN)")
    print("   3. Start with emergency fund in savings account")
    print("   4. Set up automatic transfers for systematic investing")
    print("   5. Review and rebalance portfolio annually")
    
    # Risk Warnings
    print(f"\n⚠️  IMPORTANT NOTES:")
    print("-" * 50)
    print("   • Past performance doesn't guarantee future returns")
    print("   • Diversify investments across different asset classes")
    print("   • Invest only what you can afford to lose in equity")
    print("   • Keep emergency fund separate from investments")
    print("   • Consult a certified financial advisor for large amounts")

def test_data_generation():
    """Test data generation functionality"""
    print("\n🔧 TESTING DATA GENERATION...")
    try:
        generator = EnhancedRuralDataGenerator()
        sample_data = generator.generate_ultimate_dataset(num_rows=10)
        print(f"✅ Generated sample dataset with {len(sample_data)} rows and {len(sample_data.columns)} columns")
        print("\nSample data preview:")
        print(sample_data[['Age', 'Monthly_Income_Primary', 'Risk_Tolerance', 'State']].head())
        return True
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_market_data():
    """Test market data fetching"""
    print("\n📈 TESTING MARKET DATA FETCHING...")
    try:
        fetcher = MarketDataFetcher()
        market_data = fetcher.fetch_all_market_data(use_cache=True)
        fetcher.close_connection()
        
        if market_data:
            print("✅ Market data fetched successfully!")
            if 'stocks' in market_data:
                print(f"   - Stocks: {len(market_data['stocks'])} entries")
            if 'fd_rates' in market_data:
                print(f"   - FD Rates: {len(market_data['fd_rates'])} banks")
            if 'government_schemes' in market_data:
                print(f"   - Gov Schemes: {len(market_data['government_schemes'])} schemes")
        else:
            print("⚠️  Market data fetching returned empty results")
        return True
    except Exception as e:
        print(f"❌ Market data fetching failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Starting AI Financial Advisor Demo...")
    
    # Change to correct directory
    try:
        os.chdir('RuralFinanceIntelligence (2)/RuralFinanceIntelligence')
        print("✅ Changed to project directory")
    except:
        print("⚠️  Could not change to project directory, continuing...")
    
    # Test core functionality
    data_gen_ok = test_data_generation()
    market_data_ok = test_market_data()
    
    # Create sample profile and recommendations
    print("\n🎯 GENERATING SAMPLE RECOMMENDATIONS...")
    profile = create_sample_profile()
    investment_amount = 100000  # 1 Lakh investment
    
    recommendations = simple_recommendation_logic(profile, investment_amount)
    print_recommendations(profile, recommendations, investment_amount)
    
    # Test with different investment amounts
    print("\n" + "="*60)
    print("💡 DIFFERENT INVESTMENT SCENARIOS")
    print("="*60)
    
    scenarios = [
        (25000, "Small Investment (₹25,000)"),
        (100000, "Medium Investment (₹1,00,000)"),
        (500000, "Large Investment (₹5,00,000)")
    ]
    
    for amount, title in scenarios:
        print(f"\n{title}:")
        recs = simple_recommendation_logic(profile, amount)
        for category, allocated in recs.items():
            if allocated > 0:
                percentage = (allocated / amount) * 100
                print(f"  {category.replace('_', ' ').title()}: ₹{allocated:,.0f} ({percentage:.1f}%)")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"✅ Data Generation: {'Working' if data_gen_ok else 'Failed'}")
    print(f"✅ Market Data: {'Working' if market_data_ok else 'Failed'}")
    
    print(f"\n📝 TO RUN STREAMLIT WEB INTERFACE:")
    print(f"   1. Install dependencies: pip install streamlit")
    print(f"   2. Run: streamlit run app.py --server.address=0.0.0.0 --server.port=5000")

if __name__ == "__main__":
    main()
