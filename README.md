# 🏠 California Housing Price Predictor

A professional machine learning web application that predicts California house prices using Multiple Linear Regression. Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.2-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Results](#results)

---

## 🎯 Overview

This application provides instant house price predictions for California properties using a trained Multiple Linear Regression model. The model analyzes various features including location coordinates, housing characteristics, and demographic information to estimate property values in both USD and TND.

**Why Use This App?**

- 🎯 **Accurate Predictions**: Trained on 20,640 California housing records
- ⚡ **Real-time Results**: Get instant price estimates
- 🔧 **Feature Engineering**: Automatically computes derived features
- 🗺️ **Interactive Visualization**: See property location on map
- 💱 **Multi-Currency**: View prices in USD and TND

---

## ✨ Features

- **AI-Powered Predictions**: Multiple Linear Regression model with feature engineering
- **Interactive Web Interface**: Clean, dark-themed UI built with Streamlit
- **Feature Engineering**: Automatically computes:
  - Rooms per household
  - Bedrooms per room
  - Population per household
- **Comprehensive Analysis**: Displays model metrics and property details
- **Geographic Visualization**: Interactive map showing property location
- **Model Comparison**: Compares Linear Regression, Ridge, and Lasso models

---

## 📊 Dataset

**Source**: [California Housing Dataset (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

The dataset contains information from the 1990 California census with the following features:

| Feature              | Description                          |
| -------------------- | ------------------------------------ |
| `longitude`          | Longitude coordinate                 |
| `latitude`           | Latitude coordinate                  |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms`        | Total number of rooms                |
| `total_bedrooms`     | Total number of bedrooms             |
| `population`         | Population in the district           |
| `households`         | Number of households                 |
| `median_income`      | Median income (in $10,000s)          |
| `ocean_proximity`    | Proximity to ocean (categorical)     |
| `median_house_value` | Target variable (house price)        |

**Dataset Statistics**:

- **Records**: 20,640
- **Features**: 9 input + 3 engineered
- **Target**: Median house value (USD)

---

## 🤖 Model

### Algorithm

**Multiple Linear Regression** with feature engineering and preprocessing

### Model Comparison

The project trains and compares three regression models:

1. **Linear Regression** (selected as best model)
2. **Ridge Regression** (L2 regularization)
3. **Lasso Regression** (L1 regularization)

### Feature Engineering

Three derived features are computed:

```python
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households
```

### Preprocessing Pipeline

1. **Missing Value Imputation**: Median strategy for `total_bedrooms`
2. **One-Hot Encoding**: For `ocean_proximity` categorical feature
3. **Feature Scaling**: StandardScaler on numeric features

### Performance Metrics

- **R² Score**: ~0.60 (60% variance explained)
- **MAE**: Mean Absolute Error reported
- **RMSE**: Root Mean Squared Error reported

> **Note**: R² = 0.60 is expected for linear regression on this dataset. Higher scores (0.70+) would require non-linear models like Random Forest or XGBoost.

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/KhalilAmamri/california-housing-regression.git
cd california-housing-regression
```

2. **Create virtual environment**

```bash
python -m venv .venv
```

3. **Activate virtual environment**

- Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Train the model** (if not already trained)
   - Open `notebooks/california-housing-regression.ipynb` in Jupyter/VS Code
   - Run all cells to train the model and save `best_model.pkl`

---

## 💻 Usage

### Running the Web Application

1. **Start the Streamlit app**

```bash
streamlit run app/app.py
```

2. **Access the app**

   - Open your browser to `http://localhost:8501`

3. **Make predictions**
   - Enter property details in the sidebar:
     - Location (longitude, latitude)
     - Housing characteristics (age, rooms, bedrooms)
     - Demographics (population, households, income)
     - Ocean proximity
   - Click **"Predict Price"** button
   - View results with price estimate, map, and detailed metrics

### Training the Model

To retrain the model with new data:

1. Place your dataset in `data/raw/housing.csv`
2. Open `notebooks/california-housing-regression.ipynb`
3. Run all cells sequentially
4. The best model will be saved to `models/best_model.pkl`

---

## 📁 Project Structure

```
california-housing-regression/
│
├── app/
│   └── app.py                          # Streamlit web application
│
├── data/
│   └── raw/
│       └── housing.csv                 # California housing dataset
│
├── models/
│   └── best_model.pkl                  # Trained model artifact
│
├── notebooks/
│   └── california-housing-regression.ipynb  # Model training notebook
│
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🛠️ Technologies

| Technology       | Purpose                          |
| ---------------- | -------------------------------- |
| **Python 3.8+**  | Programming language             |
| **Pandas**       | Data manipulation                |
| **NumPy**        | Numerical computing              |
| **Scikit-learn** | Machine learning algorithms      |
| **Streamlit**    | Web application framework        |
| **Plotly**       | Interactive visualizations       |
| **Joblib**       | Model serialization              |
| **Jupyter**      | Interactive notebook environment |

---

## 📈 Results

### Model Performance

- **Best Model**: Linear Regression
- **R² Score**: 0.597
- **Training Set**: 16,512 samples
- **Test Set**: 4,128 samples

### Key Insights

- Median income is the strongest predictor of house prices
- Feature engineering improves model performance
- Linear regression achieves 60% variance explanation
- Model performs consistently across train/test splits

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

**Khalil Amamri**

- GitHub: [@KhalilAmamri](https://github.com/KhalilAmamri)

---

## 🙏 Acknowledgments

- Dataset: [California Housing Prices - Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- Original Data: 1990 California Census
- Built with ❤️ using Python and Streamlit

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**
