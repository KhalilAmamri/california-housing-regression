import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")

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

# Title
st.title("🏠 California House Price Predictor")
st.markdown("### Predict house prices using Multiple Linear Regression")
st.markdown("---")

# Sidebar inputs
st.sidebar.header("📝 Enter House Features")

longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -119.5, 0.1)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0, 0.1)
housing_median_age = st.sidebar.slider("Housing Median Age", 1, 52, 25)
total_rooms = st.sidebar.slider("Total Rooms", 10, 40000, 2000, 10)
total_bedrooms = st.sidebar.slider("Total Bedrooms", 1, 6400, 400, 1)
population = st.sidebar.slider("Population", 3, 35682, 1500, 1)
households = st.sidebar.slider("Households", 1, 6082, 400, 1)
median_income = st.sidebar.slider("Median Income (in $10k)", 0.5, 15.0, 3.5, 0.1)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

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
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💵 Predicted Price")
        st.markdown(f"<h1 style='color: green;'>${prediction:,.0f}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #ff6b6b;'>{prediction_tnd:,.0f} TND</h3>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Model Details")
        st.info(f"**Model Used:** {artifact['model_name']}")
        st.success(f"**Total Features:** {len(final_columns)}")
        st.warning(f"**Engineered Features:** {len(artifact['engineered_features'])}")
    
    # Display map
    st.markdown("---")
    st.markdown("### 🗺️ Property Location")
    
    fig = go.Figure(go.Scattergeo(
        lon=[longitude],
        lat=[latitude],
        text=[f"Predicted Price: ${prediction:,.0f}"],
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            symbol='circle'
        )
    ))
    
    fig.update_geos(
        scope='usa',
        center=dict(lon=-119.5, lat=37),
        projection_scale=4,
        showland=True,
        landcolor='lightgray',
        coastlinecolor='white',
        showcountries=True
    )
    
    fig.update_layout(
        height=500,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature summary
    st.markdown("---")
    st.markdown("### 📋 Input Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rooms per Household", f"{df_input['rooms_per_household'].values[0]:.2f}")
        st.metric("Median Income", f"${median_income * 10000:,.0f}")
        st.metric("Housing Age", f"{housing_median_age} years")
    
    with col2:
        st.metric("Bedrooms per Room", f"{df_input['bedrooms_per_room'].values[0]:.2f}")
        st.metric("Total Rooms", f"{total_rooms:,}")
        st.metric("Total Bedrooms", f"{total_bedrooms:,}")
    
    with col3:
        st.metric("Population per Household", f"{df_input['population_per_household'].values[0]:.2f}")
        st.metric("Population", f"{population:,}")
        st.metric("Households", f"{households:,}")

else:
    # Welcome message
    st.info("👈 **Enter house features in the sidebar and click 'Predict Price' to get started!**")
    
    st.markdown("### 📚 About This App")
    st.markdown("""
    This app predicts California house prices using a trained **Multiple Linear Regression** model.
    
    **Features:**
    - 🎯 Predicts house prices based on 9 input features
    - 🔧 Automatically computes 3 engineered features
    - 🗺️ Shows property location on an interactive map
    - 💱 Displays price in USD and TND
    
    **Model Performance:**
    - R² Score: ~0.60 (60% variance explained)
    - Trained on 20,640 California housing records
    - Uses Linear/Ridge/Lasso regression comparison
    """)
    
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Adjust the sliders in the sidebar to match your property
    2. Select the ocean proximity category
    3. Click **'Predict Price'** button
    4. View the predicted price and location on the map
    """)
