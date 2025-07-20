# AI Financial Advisor for Rural India

## Overview

This project is a comprehensive AI-powered financial advisory system specifically designed for rural India. It combines machine learning, real-time market data, and deep understanding of rural Indian financial needs to provide personalized investment recommendations. The system is built using Streamlit for the frontend interface and incorporates advanced data generation, market data fetching, ML model training, and recommendation engines.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application
- **Layout**: Wide layout with expandable sidebar for better user experience
- **Caching**: Implements Streamlit's caching mechanisms for performance optimization
- **Session Management**: Uses session state to maintain user profiles and recommendations across interactions

### Backend Architecture
- **Data Generation**: Enhanced rural data generator with micro-personas representing different demographic segments
- **ML Pipeline**: Advanced portfolio ML trainer using ensemble methods (Random Forest, XGBoost, LightGBM)
- **Recommendation Engine**: Multi-factor recommendation system considering demographics, risk tolerance, and market conditions
- **Market Data Integration**: Real-time data fetching from Yahoo Finance and other Indian market sources

### Data Storage Solutions
- **Primary Database**: SQLite for caching market data and user sessions
- **Caching Strategy**: File-based caching for market data with TTL (30 minutes)
- **Model Storage**: Joblib for persisting trained ML models
- **Configuration**: JSON/Python dictionaries for investment products and government schemes

## Key Components

### 1. Enhanced Data Generator (`enhanced_data_generator.py`)
- Generates realistic rural Indian financial profiles using micro-personas
- Incorporates 7 parameter categories: demographics, income, assets, banking literacy, etc.
- Creates correlated data points based on research-backed regional and occupational patterns

### 2. Market Data Fetcher (`market_data_fetcher.py`)
- Fetches real-time data from Indian stock markets, mutual funds, and government schemes
- Implements SQLite caching to reduce API calls and improve performance
- Handles multiple data sources including Yahoo Finance and web scraping for Indian-specific data

### 3. ML Model Trainer (`ml_model_trainer.py`)
- Uses ensemble methods for portfolio allocation prediction
- Implements feature engineering for rural-specific parameters
- Provides model performance metrics and feature importance analysis

### 4. Recommendation Engine (`recommendation_engine.py`)
- Generates specific product recommendations with exact allocations
- Integrates market data with user profiles for personalized advice
- Considers government schemes, mutual funds, stocks, and traditional savings options

### 5. Configuration Management
- **Investment Products** (`config/investment_products.py`): Comprehensive database of investment options
- **Government Schemes** (`config/government_schemes.py`): Detailed information about central and state schemes
- **Utilities**: Data validation and financial calculations modules

## Data Flow

1. **User Input**: Collects demographic, financial, and preference data through Streamlit interface
2. **Data Validation**: Validates input using comprehensive rules in `utils/data_validation.py`
3. **Market Data Refresh**: Fetches current market data with caching for performance
4. **ML Prediction**: Uses trained models to predict optimal portfolio allocations
5. **Recommendation Generation**: Combines ML predictions with rule-based logic for specific product recommendations
6. **Visualization**: Presents recommendations through interactive Plotly charts and tables

## External Dependencies

### Python Libraries
- **UI Framework**: Streamlit for web interface
- **Data Processing**: pandas, numpy for data manipulation
- **Visualization**: plotly for interactive charts
- **ML Libraries**: scikit-learn, xgboost, lightgbm for machine learning
- **Market Data**: yfinance for stock data, requests/BeautifulSoup for web scraping
- **Database**: sqlite3 for local data storage

### Data Sources
- **Yahoo Finance**: Stock prices and basic financial data
- **Web Scraping**: Indian mutual fund NAVs and government scheme updates
- **Generated Data**: Synthetic rural user profiles based on research

### External APIs
- Yahoo Finance API for real-time stock data
- Potential integration with Indian financial data providers
- Government portal APIs for scheme information updates

## Deployment Strategy

### Development Environment
- Python 3.8+ environment with all dependencies
- SQLite database for local development and caching
- Streamlit development server for testing

### Production Considerations
- **Scaling**: Streamlit Cloud or container-based deployment
- **Database**: Consider upgrading to PostgreSQL for production scale
- **Caching**: Redis implementation for distributed caching
- **Security**: Implement user authentication and data encryption
- **Performance**: Load balancing for multiple concurrent users

### Monitoring and Maintenance
- Logging configured throughout the application for debugging
- Model performance monitoring and retraining pipelines
- Market data freshness validation
- Regular updates to government schemes and investment products

The architecture prioritizes modularity, allowing each component to be developed, tested, and deployed independently while maintaining clean interfaces between systems. The design specifically addresses the unique challenges of rural Indian financial advisory, including limited internet connectivity, varying literacy levels, and diverse financial products available in the Indian market.