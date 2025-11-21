"""
California House Price Predictor
=================================
A Streamlit web application for predicting California house prices using Multiple Linear Regression.
Trained on the California Housing Dataset (1990 census data).

Author: Khalil Amamri
Repository: california-housing-regression
"""

# ============================================================================
# IMPORTS
# ============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="California House Price Predictor", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #0f172a;
    }
    
    /* Header styling */
    h1 {
        color: #f8fafc;
        font-weight: 700;
        padding-bottom: 1rem;
    }
    
    h3 {
        color: #e2e8f0;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    h4 {
        color: #cbd5e1;
    }
    
    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #f8fafc;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] h3 {
        color: #e2e8f0;
    }
    
    /* Button styling */
    .stButton>button {
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%);
        border: none;
        color: white;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.3);
    }
    
    .price-display {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #14b8a6;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .feature-card p {
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    .feature-card strong {
        color: #f8fafc;
    }
    
    .feature-card ul {
        color: #cbd5e1;
    }
    
    .feature-card li {
        color: #cbd5e1;
    }
    
    /* Step card styling */
    .step-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 2px solid #475569;
        border-radius: 10px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .step-card:hover {
        border-color: #14b8a6;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
    
    .step-card h4 {
        color: #14b8a6;
        margin: 0.5rem 0;
    }
    
    .step-card p {
        color: #cbd5e1;
        margin-bottom: 0;
    }
    
    /* Info section styling */
    .info-section {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 5px solid #14b8a6;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .info-section h4 {
        color: #f8fafc;
        margin-top: 0;
    }
    
    .info-section p {
        color: #cbd5e1;
        margin-bottom: 0;
        line-height: 1.6;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """
    Load the trained model and preprocessing artifacts from disk.
    
    Returns:
        dict: Dictionary containing model, scaler, and other preprocessing objects
    """
    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
    
    if not model_path.exists():
        st.error(f"⚠️ Model file not found at: {model_path}")
        st.info("📝 Please run the notebook `california-housing-regression.ipynb` first to train and save the model.")
        st.stop()
    
    return joblib.load(model_path)


# Load model artifacts
artifact = load_model()
model = artifact['model']
scaler = artifact['scaler']
numeric_cols = artifact['numeric_columns']
final_columns = artifact['final_columns']
bedrooms_median = artifact['bedrooms_median']


# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown("""
<div style='background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); 
            padding: 2.5rem; 
            border-radius: 15px; 
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.5);
            border: 1px solid #334155;'>
    <h1 style='color: white; margin: 0; font-size: 3rem;'>🏠 California House Price Predictor</h1>
    <p style='color: #cbd5e1; font-size: 1.2rem; margin-top: 0.5rem;'>
        AI-Powered Price Estimation Using Multiple Linear Regression
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR - INPUT FORM
# ============================================================================
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); 
            padding: 1.5rem; 
            border-radius: 10px; 
            margin-bottom: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(20, 184, 166, 0.4);'>
    <h2 style='color: white; margin: 0;'>📝 Property Details</h2>
</div>
""", unsafe_allow_html=True)

# Location inputs
st.sidebar.markdown("### 📍 Location")
longitude = st.sidebar.slider(
    "Longitude", 
    min_value=-124.5, 
    max_value=-114.0, 
    value=-119.5, 
    step=0.1,
    help="Longitude coordinate of the property location"
)
latitude = st.sidebar.slider(
    "Latitude", 
    min_value=32.0, 
    max_value=42.0, 
    value=37.0, 
    step=0.1,
    help="Latitude coordinate of the property location"
)

# Property characteristics
st.sidebar.markdown("### 🏘️ Property Characteristics")
housing_median_age = st.sidebar.slider(
    "Housing Median Age (years)", 
    min_value=1, 
    max_value=52, 
    value=25,
    help="Median age of houses in the block"
)
total_rooms = st.sidebar.number_input(
    "Total Rooms", 
    min_value=10, 
    max_value=40000, 
    value=2000, 
    step=50,
    help="Total number of rooms in the block"
)
total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms", 
    min_value=1, 
    max_value=6400, 
    value=400, 
    step=10,
    help="Total number of bedrooms in the block"
)

# Demographics
st.sidebar.markdown("### 👥 Demographics")
population = st.sidebar.number_input(
    "Population", 
    min_value=3, 
    max_value=35682, 
    value=1500, 
    step=50,
    help="Total population in the block"
)
households = st.sidebar.number_input(
    "Households", 
    min_value=1, 
    max_value=6082, 
    value=400, 
    step=10,
    help="Total number of households in the block"
)
median_income = st.sidebar.slider(
    "Median Income ($10k)", 
    min_value=0.5, 
    max_value=15.0, 
    value=3.5, 
    step=0.1,
    help="Median income for households in the block (in tens of thousands USD)"
)

# Ocean proximity
st.sidebar.markdown("### 🌊 Proximity")
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
    help="Proximity to the ocean"
)

st.sidebar.markdown("---")

# Predict button
predict_button = st.sidebar.button(
    "🔮 Predict Price", 
    type="primary", 
    use_container_width=True
)


# ============================================================================
# PREDICTION LOGIC
# ============================================================================
if predict_button:
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    if households == 0:
        st.error("❌ Households cannot be zero!")
        st.stop()
    
    if total_rooms < total_bedrooms:
        st.error("❌ Total Bedrooms cannot exceed Total Rooms!")
        st.stop()
    
    if population < households:
        st.warning("⚠️ Warning: Population is less than Households (unusual but allowed)")
    
    # -------------------------------------------------------------------------
    # Data Preparation
    # -------------------------------------------------------------------------
    # Create input dataframe
    input_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
    
    df_input = pd.DataFrame([input_data])
    
    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    df_input['rooms_per_household'] = df_input['total_rooms'] / df_input['households']
    df_input['bedrooms_per_room'] = df_input['total_bedrooms'] / df_input['total_rooms']
    df_input['population_per_household'] = df_input['population'] / df_input['households']
    
    # -------------------------------------------------------------------------
    # Encoding and Alignment
    # -------------------------------------------------------------------------
    # One-hot encode ocean_proximity
    df_input = pd.get_dummies(df_input, columns=['ocean_proximity'], drop_first=True)
    
    # Ensure all columns match training data
    for col in final_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    df_input = df_input[final_columns]
    
    # -------------------------------------------------------------------------
    # Feature Scaling
    # -------------------------------------------------------------------------
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    # -------------------------------------------------------------------------
    # Make Prediction
    # -------------------------------------------------------------------------
    raw_prediction = model.predict(df_input)[0]
    
    # Prevent negative prices (house prices cannot be negative)
    prediction = max(raw_prediction, 0.0)
    
    # Show warning if price was clamped to zero
    if prediction == 0.0:
        st.warning("⚠️ Inputs are unrealistic or the area is extremely low-value – price set to $0")
    
    # Convert to Tunisian Dinar (1 USD = 3.15 TND)
    prediction_tnd = prediction * 3.15
    
    # -------------------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------------------
    st.markdown("### 💰 Prediction Results")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    # Price Display
    with col1:
        st.markdown(f"""
        <div class='price-display'>
            <p style='color: white; font-size: 1.2rem; margin: 0; opacity: 0.9;'>Estimated Price (USD)</p>
            <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>${prediction:,.0f}</h1>
            <p style='color: white; font-size: 1rem; margin: 0; opacity: 0.8;'>≈ {prediction_tnd:,.0f} TND</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Information
    with col2:
        st.markdown(f"""
        <div class='feature-card'>
            <h3 style='color: #14b8a6; margin-top: 0;'>🤖 Model Information</h3>
            <p><strong>Algorithm:</strong> {artifact['model_name']}</p>
            <p><strong>Total Features:</strong> {len(final_columns)}</p>
            <p><strong>Engineered Features:</strong> {len(artifact['engineered_features'])}</p>
            <p><strong>R² Score:</strong> ~0.60</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Metrics
    with col3:
        rooms_per_hh = df_input['rooms_per_household'].values[0]
        bedrooms_per_room = df_input['bedrooms_per_room'].values[0]
        pop_per_hh = df_input['population_per_household'].values[0]
        annual_income = median_income * 10000
        
        st.markdown(f"""
        <div class='feature-card'>
            <h3 style='color: #14b8a6; margin-top: 0;'>📊 Key Metrics</h3>
            <p><strong>Rooms/Household:</strong> {rooms_per_hh:.2f}</p>
            <p><strong>Bedrooms/Room:</strong> {bedrooms_per_room:.3f}</p>
            <p><strong>Population/Household:</strong> {pop_per_hh:.2f}</p>
            <p><strong>Annual Income:</strong> ${annual_income:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # Interactive Map
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🗺️ Property Location on California Map")
    
    fig = go.Figure(go.Scattergeo(
        lon=[longitude],
        lat=[latitude],
        text=[f"<b>Predicted Price:</b> ${prediction:,.0f}<br><b>Location:</b> ({latitude:.2f}, {longitude:.2f})"],
        mode='markers+text',
        marker=dict(
            size=20,
            color='#14b8a6',
            symbol='circle',
            line=dict(width=3, color='white')
        ),
        textposition='top center',
        textfont=dict(size=14, color='#f8fafc', family='Arial', weight='bold')
    ))
    
    fig.update_geos(
        scope='usa',
        center=dict(lon=-119.5, lat=37),
        projection_scale=4,
        showland=True,
        landcolor='#1e293b',
        coastlinecolor='#14b8a6',
        showcountries=True,
        countrycolor='#334155',
        showlakes=True,
        lakecolor='#0f172a'
    )
    
    fig.update_layout(
        height=500,
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------------------------------------------------------
    # Detailed Property Analysis
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Detailed Property Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Total Rooms", f"{total_rooms:,}")
        st.metric("🛏️ Total Bedrooms", f"{total_bedrooms:,}")
    
    with col2:
        st.metric("👥 Population", f"{population:,}")
        st.metric("🏘️ Households", f"{households:,}")
    
    with col3:
        st.metric("📅 Housing Age", f"{housing_median_age} yrs")
        st.metric("💵 Median Income", f"${median_income * 10000:,.0f}")
    
    with col4:
        st.metric("🏠 Rooms/House", f"{rooms_per_hh:.2f}")
        st.metric("👤 Pop/House", f"{pop_per_hh:.2f}")


# ============================================================================
# WELCOME SCREEN (No Prediction Yet)
# ============================================================================
else:
    # Welcome message
    st.markdown("""
    <div class='info-box'>
        <h2 style='color: white; margin-top: 0;'>👋 Welcome to the California House Price Predictor</h2>
        <p style='font-size: 1.1rem; line-height: 1.6;'>
            Get instant property value estimates powered by machine learning. 
            Our model analyzes multiple factors to provide accurate price predictions.
        </p>
        <p style='font-size: 1rem; margin-bottom: 0;'>
            👈 <strong>Start by entering property details in the sidebar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #14b8a6; margin-top: 0;'>🎯 Key Features</h3>
            <ul style='line-height: 2;'>
                <li><strong>AI-Powered Predictions</strong> - Multiple Linear Regression model</li>
                <li><strong>Feature Engineering</strong> - 3 automated derived features</li>
                <li><strong>Interactive Map</strong> - Visualize property location</li>
                <li><strong>Multi-Currency</strong> - Prices in USD and TND</li>
                <li><strong>Real-time Analysis</strong> - Instant results</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #14b8a6; margin-top: 0;'>📊 Model Performance</h3>
            <ul style='line-height: 2;'>
                <li><strong>R² Score:</strong> ~0.60 (60% variance explained)</li>
                <li><strong>Training Data:</strong> 20,640 California properties</li>
                <li><strong>Features Used:</strong> 9 input + 3 engineered</li>
                <li><strong>Model Type:</strong> Linear/Ridge/Lasso comparison</li>
                <li><strong>Accuracy:</strong> Production-ready performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use guide
    st.markdown("---")
    st.markdown("### 🚀 How to Use This App")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='step-card'>
            <div style='font-size: 3rem; text-align: center;'>1️⃣</div>
            <h4 style='text-align: center;'>Enter Details</h4>
            <p style='text-align: center;'>Fill in property information in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='step-card'>
            <div style='font-size: 3rem; text-align: center;'>2️⃣</div>
            <h4 style='text-align: center;'>Select Proximity</h4>
            <p style='text-align: center;'>Choose ocean proximity category</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='step-card'>
            <div style='font-size: 3rem; text-align: center;'>3️⃣</div>
            <h4 style='text-align: center;'>Predict</h4>
            <p style='text-align: center;'>Click the prediction button</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='step-card'>
            <div style='font-size: 3rem; text-align: center;'>4️⃣</div>
            <h4 style='text-align: center;'>View Results</h4>
            <p style='text-align: center;'>Get instant price estimates and analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional information
    st.markdown("---")
    st.markdown("""
    <div class='info-section'>
        <h4>ℹ️ About the Dataset</h4>
        <p>
            This model is trained on the <strong>California Housing Dataset</strong>, which contains information from 
            the 1990 California census. The dataset includes median house values, location coordinates, housing characteristics, 
            and demographic information for California districts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-section'>
        <h4>🔍 Model Details</h4>
        <p>
            The prediction model uses <strong>Multiple Linear Regression</strong> with engineered features to estimate house prices. 
            The model achieves an R² score of approximately <strong>0.60</strong>, meaning it explains 60% of the variance in house prices. 
            This is considered reasonable performance for linear models on real-world housing data.
        </p>
    </div>
    """, unsafe_allow_html=True)

