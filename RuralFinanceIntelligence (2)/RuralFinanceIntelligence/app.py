import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import logging

# Import our modules
from enhanced_data_generator import EnhancedRuralDataGenerator
from market_data_fetcher import MarketDataFetcher
from ml_model_trainer import AdvancedPortfolioMLTrainer
from recommendation_engine import AdvancedRecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Financial Advisor for Rural India",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_market_data():
    """Load market data with caching"""
    try:
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_all_market_data(use_cache=True)
        fetcher.close_connection()
        return data
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")
        return None

@st.cache_resource
def load_recommendation_engine():
    """Load recommendation engine with caching"""
    try:
        # Check if trained models exist
        model_files = [f for f in os.listdir('.') if f.startswith('trained_portfolio_models_') and f.endswith('.joblib')]
        
        if model_files:
            # Use the latest model file
            latest_model = max(model_files, key=lambda x: os.path.getctime(x))
            engine = AdvancedRecommendationEngine(model_path=latest_model)
            st.success(f"✅ Loaded trained ML models from {latest_model}")
        else:
            engine = AdvancedRecommendationEngine()
            st.warning("⚠️ No trained models found. Using rule-based recommendations.")
        
        return engine
    except Exception as e:
        st.error(f"Error loading recommendation engine: {str(e)}")
        return None

def main():
    """Main application function"""
    
    # Header
    st.title("🏦 AI Financial Advisor for Rural India")
    st.markdown("### Personalized Investment Recommendations Based on Advanced ML Models")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Profile Input", "Market Data", "Recommendations", "Model Training", "Data Generation"]
    )
    
    if page == "Profile Input":
        profile_input_page()
    elif page == "Market Data":
        market_data_page()
    elif page == "Recommendations":
        recommendations_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Data Generation":
        data_generation_page()

def profile_input_page():
    """User profile input page"""
    st.header("👤 User Profile Input")
    st.markdown("Please provide your details for personalized financial recommendations.")
    
    # Create tabs for different parameter categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Demographics", "Income & Employment", "Assets & Liabilities", 
        "Banking & Literacy", "Financial Goals", "Location & Seasonal", "Government Schemes"
    ])
    
    with tab1:
        st.subheader("🧑‍🌾 Demographic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Widowed", "Divorced"])
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
        
        with col2:
            education = st.selectbox("Education Level", ["Illiterate", "Primary", "Secondary", "Graduate", "Post Graduate"])
            community = st.selectbox("Community", ["General", "OBC", "SC", "ST"])
            occupation = st.selectbox("Primary Occupation", [
                "Marginal Farmer", "Irrigated Farmer", "Daily Wage Laborer", "Construction Worker",
                "Small Business Owner", "Trader", "Shop Owner", "Delivery Partner", "Cab Driver",
                "Government Employee", "Private Sector Employee", "Teacher", "Retired"
            ])
        
        st.session_state.user_profile.update({
            'Age': age, 'Gender': gender, 'Marital_Status': marital_status,
            'Dependents': dependents, 'Education_Level': education,
            'Community': community, 'Occupation': occupation
        })
    
    with tab2:
        st.subheader("💰 Income and Employment")
        col1, col2 = st.columns(2)
        
        with col1:
            primary_income = st.number_input("Monthly Primary Income (₹)", min_value=0, value=25000)
            secondary_income = st.number_input("Monthly Secondary Income (₹)", min_value=0, value=5000)
            income_stability = st.selectbox("Income Stability", ["Fixed", "Stable", "Seasonal-Stable", "Variable", "Seasonal"])
        
        with col2:
            total_annual = (primary_income + secondary_income) * 12
            st.metric("Total Annual Income", f"₹{total_annual:,.0f}")
            
        st.session_state.user_profile.update({
            'Monthly_Income_Primary': primary_income,
            'Monthly_Income_Secondary': secondary_income,
            'Income_Stability': income_stability
        })
    
    with tab3:
        st.subheader("🏡 Assets and Liabilities")
        col1, col2 = st.columns(2)
        
        with col1:
            land_acres = st.number_input("Land Owned (Acres)", min_value=0.0, value=2.0, step=0.5)
            livestock_value = st.number_input("Livestock Value (₹)", min_value=0, value=25000)
            gold_grams = st.number_input("Gold/Silver Owned (Grams)", min_value=0.0, value=50.0, step=5.0)
            vehicle = st.selectbox("Vehicle Owned", ["None", "Bicycle", "Motorcycle", "Car", "Tractor"])
        
        with col2:
            debt_amount = st.number_input("Total Debt Amount (₹)", min_value=0, value=100000)
            debt_source = st.selectbox("Primary Debt Source", ["Bank", "SHG", "Moneylender", "Family", "Multiple"])
            debt_rate = st.number_input("Average Debt Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0)
        
        # Calculate derived metrics
        total_assets = land_acres * 200000 + livestock_value + gold_grams * 5000
        debt_to_income = debt_amount / (total_annual + 1) if total_annual > 0 else 0
        
        st.session_state.user_profile.update({
            'Land_Owned_Acres': land_acres, 'Livestock_Value': livestock_value,
            'Gold_Grams': gold_grams, 'Vehicle_Owned': vehicle,
            'Total_Debt_Amount': debt_amount, 'Debt_Source': debt_source,
            'Debt_Interest_Rate': debt_rate, 'Total_Asset_Value': total_assets,
            'Debt_to_Income_Ratio': debt_to_income
        })
    
    with tab4:
        st.subheader("🏦 Banking and Financial Literacy")
        col1, col2 = st.columns(2)
        
        with col1:
            bank_account = st.selectbox("Bank Account Type", ["Jan Dhan", "Basic Savings", "Regular Savings", "Current"])
            bank_distance = st.number_input("Distance to Bank (KM)", min_value=0.0, value=5.0, step=0.5)
            digital_literacy = st.selectbox("Digital Literacy", ["Low", "Low-Medium", "Medium", "Medium-High", "High"])
            financial_literacy = st.slider("Financial Literacy Score (1-10)", 1, 10, 6)
        
        with col2:
            smartphone = st.checkbox("Smartphone Access", value=True)
            internet = st.checkbox("Internet Access", value=True)
            upi_usage = st.checkbox("UPI Usage", value=True)
            scheme_awareness = st.slider("Government Scheme Awareness (1-10)", 1, 10, 5)
        
        st.session_state.user_profile.update({
            'Bank_Account_Type': bank_account, 'Bank_Distance_KM': bank_distance,
            'Digital_Literacy': digital_literacy, 'Financial_Literacy_Score': financial_literacy,
            'Smartphone_Access': smartphone, 'Internet_Access': internet,
            'UPI_Usage': upi_usage, 'Govt_Scheme_Awareness': scheme_awareness
        })
    
    with tab5:
        st.subheader("🎯 Financial Goals and Risk Appetite")
        col1, col2 = st.columns(2)
        
        with col1:
            short_goal = st.selectbox("Short Term Goal", [
                "Emergency Fund", "Loan Repayment", "Medical Expenses",
                "Education Fees", "Home Repair", "Festival Expenses"
            ])
            short_horizon = st.number_input("Short Term Horizon (Months)", min_value=1, max_value=60, value=12)
            risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
        
        with col2:
            long_goal = st.selectbox("Long Term Goal", [
                "House Purchase", "Daughter Marriage", "Tractor Purchase",
                "Land Purchase", "Children Education", "Retirement"
            ])
            long_horizon = st.number_input("Long Term Horizon (Years)", min_value=1, max_value=50, value=10)
            investment_preference = st.selectbox("Investment Preference", ["FD", "Savings", "Gold", "PPF", "Mutual_Funds", "Stocks"])
        
        st.session_state.user_profile.update({
            'Short_Term_Goal': short_goal, 'Short_Term_Horizon_Months': short_horizon,
            'Long_Term_Goal': long_goal, 'Long_Term_Horizon_Years': long_horizon,
            'Risk_Tolerance': risk_tolerance, 'Investment_Preference': investment_preference
        })
    
    with tab6:
        st.subheader("📅 Location and Seasonal Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            state = st.selectbox("State", [
                "Andhra Pradesh", "Bihar", "Gujarat", "Haryana", "Karnataka",
                "Kerala", "Madhya Pradesh", "Maharashtra", "Punjab", "Rajasthan",
                "Tamil Nadu", "Telangana", "Uttar Pradesh", "West Bengal"
            ])
            district = st.text_input("District", value="District_1")
            primary_crop = st.selectbox("Primary Crop (if farmer)", ["NA", "Rice", "Wheat", "Cotton", "Sugarcane", "Soybean"])
        
        with col2:
            electricity_hours = st.number_input("Electricity Hours/Day", min_value=0, max_value=24, value=18)
            water_access = st.selectbox("Water Access", ["Tap", "Borewell", "Well", "Community"])
            road_connectivity = st.selectbox("Road Connectivity", ["Excellent", "Good", "Average", "Poor"])
        
        st.session_state.user_profile.update({
            'State': state, 'District': district, 'Primary_Crop': primary_crop,
            'Electricity_Hours_Daily': electricity_hours, 'Water_Access': water_access,
            'Road_Connectivity': road_connectivity
        })
    
    with tab7:
        st.subheader("✅ Government Scheme Eligibility")
        st.markdown("These will be auto-calculated based on your profile, but you can override if needed.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pm_kisan = st.checkbox("PM-Kisan Eligible", value=(land_acres > 0 and total_annual < 200000))
            pmjjby = st.checkbox("PMJJBY Eligible", value=(18 <= age <= 50))
            pmsby = st.checkbox("PMSBY Eligible", value=(18 <= age <= 70))
        
        with col2:
            apy_eligible = st.checkbox("APY Eligible", value=(18 <= age <= 40))
            bpl_status = st.checkbox("BPL Status", value=(total_annual < 120000))
            aadhaar_linked = st.checkbox("Aadhaar Linked Bank Account", value=True)
        
        st.session_state.user_profile.update({
            'PM_Kisan_Eligible': pm_kisan, 'PMJJBY_Eligible': pmjjby,
            'PMSBY_Eligible': pmsby, 'APY_Eligible': apy_eligible,
            'BPL_Status': bpl_status, 'Aadhaar_Linked_Bank': aadhaar_linked
        })
    
    # Summary and validation
    st.header("📊 Profile Summary")
    if st.session_state.user_profile:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Age", f"{age} years")
            st.metric("Monthly Income", f"₹{primary_income + secondary_income:,.0f}")
        
        with col2:
            st.metric("Risk Tolerance", risk_tolerance)
            st.metric("Financial Literacy", f"{financial_literacy}/10")
        
        with col3:
            st.metric("Total Assets", f"₹{total_assets:,.0f}")
            st.metric("Debt Ratio", f"{debt_to_income:.2f}")
        
        with col4:
            st.metric("Bank Distance", f"{bank_distance} km")
            st.metric("Digital Access", "Yes" if smartphone and internet else "Limited")
        
        if st.button("✅ Validate & Save Profile", type="primary"):
            st.success("✅ Profile saved successfully! Go to 'Recommendations' to get personalized advice.")
            st.balloons()

def market_data_page():
    """Market data display page"""
    st.header("📈 Real-Time Market Data")
    
    # Load market data
    if st.button("🔄 Refresh Market Data"):
        st.session_state.market_data = None
    
    if st.session_state.market_data is None:
        with st.spinner("Loading market data..."):
            st.session_state.market_data = load_market_data()
    
    if st.session_state.market_data:
        market_data = st.session_state.market_data
        
        # Tabs for different data types
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Stocks", "🏦 FD Rates", "🎯 Mutual Funds", "🏛️ Government Schemes"])
        
        with tab1:
            st.subheader("Top NSE Stocks")
            stocks = market_data.get('stocks', {})
            
            if stocks:
                stock_df = []
                for symbol, data in stocks.items():
                    if isinstance(data, dict):
                        stock_df.append({
                            'Symbol': symbol,
                            'Company': data.get('company_name', symbol.replace('.NS', '')),
                            'Price (₹)': data.get('current_price', 0),
                            'Change (%)': data.get('change_percent', 0),
                            'Sector': data.get('sector', 'Unknown')
                        })
                
                if stock_df:
                    df = pd.DataFrame(stock_df)
                    st.dataframe(df, use_container_width=True)
                    
                    # Create a chart
                    fig = px.bar(df.head(10), x='Company', y='Change (%)', 
                               title="Top 10 Stocks Performance Today",
                               color='Change (%)', color_continuous_scale='RdYlGn')
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stock data available. Market data fetcher may need configuration.")
        
        with tab2:
            st.subheader("Fixed Deposit Rates")
            fd_rates = market_data.get('fd_rates', {})
            
            if fd_rates:
                fd_df = []
                for bank, rates in fd_rates.items():
                    if isinstance(rates, list):
                        for rate_info in rates:
                            fd_df.append({
                                'Bank': bank,
                                'Tenure (Months)': rate_info.get('tenure_months', 0),
                                'Rate (%)': rate_info.get('rate', 0),
                                'Senior Rate (%)': rate_info.get('senior_citizen_rate', 0),
                                'Min Amount (₹)': rate_info.get('min_amount', 0),
                                'Bank Type': rate_info.get('bank_type', 'Unknown')
                            })
                
                if fd_df:
                    df = pd.DataFrame(fd_df)
                    st.dataframe(df, use_container_width=True)
                    
                    # Best rates chart
                    best_rates = df.groupby('Bank')['Rate (%)'].max().sort_values(ascending=False).head(10)
                    fig = px.bar(x=best_rates.index, y=best_rates.values,
                               title="Top 10 Banks - Best FD Rates",
                               labels={'x': 'Bank', 'y': 'Best Rate (%)'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No FD rates available.")
        
        with tab3:
            st.subheader("Mutual Funds")
            mf_data = market_data.get('mutual_funds', {})
            
            if mf_data:
                for category, funds in mf_data.items():
                    st.write(f"**{category} Funds:**")
                    if funds:
                        mf_df = pd.DataFrame(funds)
                        st.dataframe(mf_df[['scheme_name', 'nav', 'returns_1y', 'returns_3y', 'expense_ratio']], 
                                   use_container_width=True)
            else:
                st.info("No mutual fund data available.")
        
        with tab4:
            st.subheader("Government Schemes")
            govt_schemes = market_data.get('government_schemes', {})
            
            if govt_schemes:
                scheme_df = []
                for scheme, details in govt_schemes.items():
                    scheme_df.append({
                        'Scheme': scheme,
                        'Full Name': details.get('full_name', scheme),
                        'Interest Rate (%)': details.get('interest_rate', 'N/A'),
                        'Lock-in (Years)': details.get('lock_in_period_years', 'N/A'),
                        'Min Investment (₹)': details.get('min_investment', 'N/A'),
                        'Tax Benefit': details.get('tax_benefit', 'N/A')
                    })
                
                df = pd.DataFrame(scheme_df)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No government scheme data available.")
    else:
        st.error("❌ Failed to load market data. Please check your internet connection.")

def recommendations_page():
    """Recommendations page"""
    st.header("🎯 Personalized Investment Recommendations")
    
    if not st.session_state.user_profile:
        st.warning("⚠️ Please complete your profile in the 'Profile Input' section first.")
        return
    
    # Investment amount input
    st.subheader("💰 Investment Amount")
    col1, col2 = st.columns(2)
    
    with col1:
        investment_amount = st.number_input(
            "How much would you like to invest? (₹)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=5000
        )
    
    with col2:
        investment_type = st.selectbox(
            "Investment Type",
            ["One-time Investment", "Monthly SIP", "Quarterly Investment"]
        )
    
    # Generate recommendations
    if st.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Analyzing your profile and generating personalized recommendations..."):
            try:
                engine = load_recommendation_engine()
                if engine:
                    recommendations = engine.generate_specific_recommendations(
                        st.session_state.user_profile, 
                        investment_amount
                    )
                    st.session_state.recommendations = recommendations
                else:
                    st.error("❌ Failed to load recommendation engine.")
                    return
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                return
    
    # Display recommendations
    if st.session_state.recommendations:
        recommendations = st.session_state.recommendations
        
        # Summary metrics
        st.subheader("📊 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        summary = recommendations['summary']
        with col1:
            st.metric("Expected Annual Return", f"{summary['expected_annual_return']:.1f}%")
        with col2:
            st.metric("Risk Level", summary['risk_level'])
        with col3:
            st.metric("Tax Benefits", f"₹{summary['tax_benefits']:,.0f}")
        with col4:
            st.metric("Liquidity Score", f"{summary['liquidity_score']:.1f}/5")
        
        # Allocation chart
        st.subheader("🥧 Portfolio Allocation")
        allocations = recommendations['allocations']
        
        # Filter out zero allocations for cleaner visualization
        non_zero_allocations = {k: v for k, v in allocations.items() if v > 0.01}
        
        if non_zero_allocations:
            fig = px.pie(
                values=list(non_zero_allocations.values()),
                names=[k.replace('_', ' ') for k in non_zero_allocations.keys()],
                title="Portfolio Allocation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Specific products
        st.subheader("📋 Specific Product Recommendations")
        products = recommendations['specific_products']
        
        for i, product in enumerate(products, 1):
            with st.expander(f"{i}. {product['name']} - ₹{product['amount']:,.0f} ({product['allocation_percentage']:.1f}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Type:** {product['type']}")
                    st.write(f"**Category:** {product['category']}")
                    st.write(f"**Amount:** ₹{product['amount']:,.0f}")
                    st.write(f"**Expected Return:** {product['expected_return']:.1f}%")
                    st.write(f"**Risk Level:** {product['risk_level']}")
                
                with col2:
                    st.write(f"**Liquidity:** {product['liquidity']}")
                    if 'tax_benefit' in product:
                        st.write(f"**Tax Benefit:** {product['tax_benefit']}")
                    if 'tax_implication' in product:
                        st.write(f"**Tax Implication:** {product['tax_implication']}")
                    if 'min_investment' in product:
                        st.write(f"**Min Investment:** ₹{product['min_investment']:,.0f}")
                
                # Product-specific details
                if product['type'] == 'Equity':
                    st.write(f"**Shares:** {product.get('shares', 0)} @ ₹{product.get('price_per_share', 0):.2f}")
                    st.write(f"**Sector:** {product.get('sector', 'Unknown')}")
                elif product['type'] == 'Fixed Deposit':
                    st.write(f"**Bank:** {product.get('bank', 'N/A')}")
                    st.write(f"**Tenure:** {product.get('tenure_months', 0)} months")
                    st.write(f"**Maturity Amount:** ₹{product.get('maturity_amount', 0):,.0f}")
                elif 'ELSS' in product['type']:
                    st.write(f"**NAV:** ₹{product.get('nav', 0):.2f}")
                    st.write(f"**Units:** {product.get('units', 0):.3f}")
                    st.write(f"**Lock-in:** {product.get('lock_in_period', 'N/A')}")
        
        # Eligibility alerts
        if recommendations['eligibility_alerts']:
            st.subheader("🔔 Eligibility Alerts")
            for alert in recommendations['eligibility_alerts']:
                st.info(alert)
        
        # Accessibility warnings
        if recommendations['accessibility_warnings']:
            st.subheader("⚠️ Accessibility Considerations")
            for warning in recommendations['accessibility_warnings']:
                st.warning(warning)
        
        # Action plan
        st.subheader("📝 Action Plan")
        st.markdown("""
        **Next Steps:**
        1. Review the recommended portfolio allocation above
        2. Start with government schemes and low-risk instruments
        3. Gradually move to equity investments as you gain experience
        4. Set up SIPs for mutual funds to benefit from rupee cost averaging
        5. Review and rebalance your portfolio annually
        
        **Important Notes:**
        - Invest only in products you understand
        - Diversification is key to managing risk
        - Start small and increase investments gradually
        - Keep emergency funds separate from investments
        """)

def model_training_page():
    """Model training page"""
    st.header("🤖 ML Model Training")
    st.markdown("Train advanced machine learning models on rural financial data.")
    
    # Dataset status
    st.subheader("📊 Dataset Status")
    dataset_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'dataset' in f.lower()]
    
    if dataset_files:
        latest_dataset = max(dataset_files, key=lambda x: os.path.getctime(x))
        try:
            df = pd.read_csv(latest_dataset)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dataset File", latest_dataset)
            with col2:
                st.metric("Total Rows", f"{len(df):,}")
            with col3:
                st.metric("Features", len(df.columns))
            
            st.success(f"✅ Dataset loaded: {latest_dataset}")
            
            # Dataset preview
            if st.checkbox("Show Dataset Preview"):
                st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        st.warning("⚠️ No dataset found. Please generate data first.")
    
    # Model training
    st.subheader("🎯 Model Training")
    
    if dataset_files:
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_to_use = st.selectbox("Select Dataset", dataset_files)
            training_params = st.expander("Training Parameters")
            
            with training_params:
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
                n_estimators = st.slider("Number of Estimators", 50, 500, 200)
                max_depth = st.slider("Max Depth", 3, 20, 6)
        
        with col2:
            st.info("""
            **Training Process:**
            1. Feature engineering and preprocessing
            2. Portfolio allocation target creation
            3. Train ensemble models (XGBoost, Random Forest, etc.)
            4. Model evaluation and selection
            5. Save trained models
            """)
        
        if st.button("🚀 Start Training", type="primary"):
            if not dataset_to_use:
                st.error("Please select a dataset.")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize trainer
                status_text.text("Initializing ML trainer...")
                progress_bar.progress(10)
                
                trainer = AdvancedPortfolioMLTrainer()
                
                # Train models
                status_text.text("Training models... This may take several minutes.")
                progress_bar.progress(30)
                
                trainer.train_complete_system(dataset_to_use)
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                st.success("✅ Models trained successfully!")
                st.balloons()
                
                # Display training results
                if hasattr(trainer, 'model_performance'):
                    st.subheader("📈 Training Results")
                    
                    performance_data = []
                    for target, models in trainer.model_performance.items():
                        for model_name, metrics in models.items():
                            performance_data.append({
                                'Target': target,
                                'Model': model_name,
                                'R² Score': metrics['test_r2'],
                                'MAE': metrics['test_mae'],
                                'RMSE': metrics['test_rmse']
                            })
                    
                    if performance_data:
                        perf_df = pd.DataFrame(performance_data)
                        st.dataframe(perf_df, use_container_width=True)
                        
                        # Best model by target
                        best_models = perf_df.groupby('Target')['R² Score'].max()
                        fig = px.bar(x=best_models.index, y=best_models.values,
                                   title="Best Model Performance by Asset Class",
                                   labels={'x': 'Asset Class', 'y': 'R² Score'})
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                progress_bar.progress(0)
                status_text.text("Training failed.")
    
    # Model status
    st.subheader("🎯 Trained Models")
    model_files = [f for f in os.listdir('.') if f.startswith('trained_portfolio_models_') and f.endswith('.joblib')]
    
    if model_files:
        for model_file in sorted(model_files, reverse=True):
            creation_time = datetime.fromtimestamp(os.path.getctime(model_file))
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(model_file)
            with col2:
                st.text(f"{file_size:.1f} MB")
            with col3:
                st.text(creation_time.strftime("%Y-%m-%d %H:%M"))
    else:
        st.info("No trained models found.")

def data_generation_page():
    """Data generation page"""
    st.header("📊 Dataset Generation")
    st.markdown("Generate comprehensive rural financial datasets for model training.")
    
    # Generation parameters
    st.subheader("⚙️ Generation Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        num_rows = st.selectbox(
            "Number of Profiles",
            [10000, 50000, 100000, 500000, 1000000, 10000000],
            index=2
        )
        
        include_personas = st.multiselect(
            "Include Personas",
            [
                "Marginal_Farmer_Bihar", "Irrigated_Farmer_Punjab", "Rural_Laborer_UP",
                "Small_Town_Trader_MH", "Urban_Gig_Worker_KA", "Salaried_Formal_TN", "Retired_Pensioner"
            ],
            default=[
                "Marginal_Farmer_Bihar", "Irrigated_Farmer_Punjab", "Rural_Laborer_UP",
                "Small_Town_Trader_MH", "Urban_Gig_Worker_KA", "Salaried_Formal_TN", "Retired_Pensioner"
            ]
        )
    
    with col2:
        output_format = st.selectbox("Output Format", ["CSV", "Parquet"])
        filename_prefix = st.text_input("Filename Prefix", value="rural_financial_dataset")
        
        st.info(f"""
        **Estimated Generation Time:**
        - 100K profiles: ~2-3 minutes
        - 1M profiles: ~15-20 minutes
        - 10M profiles: ~2-3 hours
        
        **Dataset Features:**
        - 50+ engineered features
        - Realistic correlations
        - Government scheme eligibility
        - Asset-liability calculations
        """)
    
    # Generation process
    if st.button("🚀 Generate Dataset", type="primary"):
        if not include_personas:
            st.error("Please select at least one persona.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Initialize generator
            status_text.text("Initializing data generator...")
            progress_bar.progress(10)
            
            generator = EnhancedRuralDataGenerator()
            
            # Generate dataset
            status_text.text(f"Generating {num_rows:,} profiles... This may take several minutes.")
            progress_bar.progress(30)
            
            dataset = generator.generate_ultimate_dataset(num_rows=num_rows)
            
            progress_bar.progress(70)
            status_text.text("Saving dataset...")
            
            # Save dataset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{filename_prefix}_{timestamp}.{output_format.lower()}"
            
            if output_format == "CSV":
                dataset.to_csv(output_filename, index=False)
            else:
                dataset.to_parquet(output_filename, index=False)
            
            progress_bar.progress(100)
            status_text.text("Dataset generation completed!")
            
            st.success(f"✅ Dataset generated successfully!")
            st.balloons()
            
            # Display dataset summary
            st.subheader("📊 Dataset Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Profiles", f"{len(dataset):,}")
            with col2:
                st.metric("Features", len(dataset.columns))
            with col3:
                st.metric("File Size", f"{os.path.getsize(output_filename) / (1024*1024):.1f} MB")
            with col4:
                st.metric("Output File", output_filename)
            
            # Preview
            if st.checkbox("Show Dataset Preview"):
                st.dataframe(dataset.head(), use_container_width=True)
                
                # Basic statistics
                st.subheader("📈 Basic Statistics")
                numeric_cols = dataset.select_dtypes(include=[np.number]).columns
                st.dataframe(dataset[numeric_cols].describe(), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Dataset generation failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("Generation failed.")
    
    # Existing datasets
    st.subheader("📁 Existing Datasets")
    dataset_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.parquet')) and 'dataset' in f.lower()]
    
    if dataset_files:
        for dataset_file in sorted(dataset_files, reverse=True):
            creation_time = datetime.fromtimestamp(os.path.getctime(dataset_file))
            file_size = os.path.getsize(dataset_file) / (1024 * 1024)  # MB
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.text(dataset_file)
            with col2:
                st.text(f"{file_size:.1f} MB")
            with col3:
                st.text(creation_time.strftime("%Y-%m-%d %H:%M"))
            with col4:
                if st.button(f"Load", key=f"load_{dataset_file}"):
                    try:
                        if dataset_file.endswith('.csv'):
                            preview_df = pd.read_csv(dataset_file, nrows=5)
                        else:
                            preview_df = pd.read_parquet(dataset_file).head()
                        
                        st.dataframe(preview_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error loading {dataset_file}: {str(e)}")
    else:
        st.info("No datasets found. Generate your first dataset above.")

if __name__ == "__main__":
    main()
