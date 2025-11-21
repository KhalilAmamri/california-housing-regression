import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="California House Price Predictor", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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

# Load the trained model and preprocessing artifacts
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
    if not model_path.exists():
        st.error(f"⚠️ Model file not found at: {model_path}")
        st.info("📝 Please run the notebook `california-housing-regression.ipynb` first to train and save the model.")
        st.stop()
    return joblib.load(model_path)

artifact = load_model()
model = artifact['model']
scaler = artifact['scaler']
numeric_cols = artifact['numeric_columns']
final_columns = artifact['final_columns']
bedrooms_median = artifact['bedrooms_median']

# Header with gradient background
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

# Sidebar inputs with enhanced styling
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

st.sidebar.markdown("### 📍 Location")
longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -119.5, 0.1)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0, 0.1)

st.sidebar.markdown("### 🏘️ Property Characteristics")
housing_median_age = st.sidebar.slider("Housing Median Age (years)", 1, 52, 25)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=10, max_value=40000, value=2000, step=50)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1, max_value=6400, value=400, step=10)

st.sidebar.markdown("### 👥 Demographics")
population = st.sidebar.number_input("Population", min_value=3, max_value=35682, value=1500, step=50)
households = st.sidebar.number_input("Households", min_value=1, max_value=6082, value=400, step=10)
median_income = st.sidebar.slider("Median Income ($10k)", 0.5, 15.0, 3.5, 0.1)

st.sidebar.markdown("### 🌊 Proximity")
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

st.sidebar.markdown("---")
# Predict button
predict_button = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)

if predict_button:
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
    
    # Create engineered features
    df_input['rooms_per_household'] = df_input['total_rooms'] / df_input['households']
    df_input['bedrooms_per_room'] = df_input['total_bedrooms'] / df_input['total_rooms']
    df_input['population_per_household'] = df_input['population'] / df_input['households']
    
    # One-hot encode ocean_proximity
    df_input = pd.get_dummies(df_input, columns=['ocean_proximity'], drop_first=True)
    
    # Ensure all columns match training data
    for col in final_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    df_input = df_input[final_columns]
    
    # Scale numeric features
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])
    
    # Make prediction
    prediction = model.predict(df_input)[0]
    prediction_tnd = prediction * 3.15  # Convert to Tunisian Dinar
    
    # Display results with enhanced design
    st.markdown("### 💰 Prediction Results")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.markdown("""
        <div class='price-display'>
            <p style='color: white; font-size: 1.2rem; margin: 0; opacity: 0.9;'>Estimated Price (USD)</p>
            <h1 style='color: white; font-size: 3rem; margin: 0.5rem 0;'>${:,.0f}</h1>
            <p style='color: white; font-size: 1rem; margin: 0; opacity: 0.8;'>≈ {:,.0f} TND</p>
        </div>
        """.format(prediction, prediction_tnd), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #14b8a6; margin-top: 0;'>🤖 Model Information</h3>
            <p><strong>Algorithm:</strong> {}</p>
            <p><strong>Total Features:</strong> {}</p>
            <p><strong>Engineered Features:</strong> {}</p>
            <p><strong>R² Score:</strong> ~0.60</p>
        </div>
        """.format(artifact['model_name'], len(final_columns), len(artifact['engineered_features'])), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3 style='color: #14b8a6; margin-top: 0;'>📊 Key Metrics</h3>
            <p><strong>Rooms/Household:</strong> {:.2f}</p>
            <p><strong>Bedrooms/Room:</strong> {:.3f}</p>
            <p><strong>Population/Household:</strong> {:.2f}</p>
            <p><strong>Annual Income:</strong> ${:,.0f}</p>
        </div>
        """.format(
            df_input['rooms_per_household'].values[0],
            df_input['bedrooms_per_room'].values[0],
            df_input['population_per_household'].values[0],
            median_income * 10000
        ), unsafe_allow_html=True)
    
    # Display map with enhanced styling
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
    
    # Feature summary with cards
    st.markdown("---")
    st.markdown("### 📋 Detailed Property Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Total Rooms", f"{total_rooms:,}", delta=None)
        st.metric("🛏️ Total Bedrooms", f"{total_bedrooms:,}", delta=None)
    
    with col2:
        st.metric("👥 Population", f"{population:,}", delta=None)
        st.metric("🏘️ Households", f"{households:,}", delta=None)
    
    with col3:
        st.metric("📅 Housing Age", f"{housing_median_age} yrs", delta=None)
        st.metric("💵 Median Income", f"${median_income * 10000:,.0f}", delta=None)
    
    with col4:
        st.metric("🏠 Rooms/House", f"{df_input['rooms_per_household'].values[0]:.2f}", delta=None)
        st.metric("👤 Pop/House", f"{df_input['population_per_household'].values[0]:.2f}", delta=None)

else:
    # Welcome message with enhanced design
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
    
    # How to use
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
    
    # Additional info
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
