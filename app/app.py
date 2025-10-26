"""
Air Quality Analysis & Prediction Application (Optimized for Speed & Reliability)
Streamlit web app for exploring urban air quality data and predicting PM2.5 levels
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Page config
st.set_page_config(
    page_title="Air Quality Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Helpers --------------------
def safe_get(meta, key, default="N/A"):
    return meta.get(key, default) if isinstance(meta, dict) else default

def ensure_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

def most_frequent_label(series):
    if series.isnull().all():
        return None
    return series.mode().iat[0]

# -------------------- Data Loaders --------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/air_quality_global.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def load_metadata():
    try:
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        # don't call st.error here (caching context) — return None and handle later
        return None

# -------------------- Preprocessing --------------------
def preprocess_data(df):
    df = df.copy()
    # Basic column checks
    required_cols = ['city', 'country', 'latitude', 'longitude', 'year', 'month', 'pm25_ugm3', 'no2_ugm3']
    missing = ensure_columns(df, required_cols)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Missing value imputation (median for numeric)
    df['pm25_ugm3'] = df['pm25_ugm3'].fillna(df['pm25_ugm3'].median())
    df['no2_ugm3'] = df['no2_ugm3'].fillna(df['no2_ugm3'].median())
    df['latitude'] = df['latitude'].fillna(df['latitude'].median())
    df['longitude'] = df['longitude'].fillna(df['longitude'].median())

    # Data quality filter (if column exists)
    if 'data_quality' in df.columns:
        # keep reasonable flags if present, otherwise don't filter aggressively
        valid_flags = ['Good', 'Moderate', 'Fair']
        df = df[df['data_quality'].isin(valid_flags) | df['data_quality'].isna()]

    # Ensure month is integer and in 1..12
    df['month'] = df['month'].astype(int).clip(1, 12)

    # Create season
    df['season'] = df['month'].apply(lambda x:
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall'
    )

    # PM2.5 category (bins chosen for illustrative purpose)
    df['pm25_category'] = pd.cut(df['pm25_ugm3'],
                                  bins=[-1, 12, 35.4, 55.4, 150.4, 500],
                                  labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy'])
    return df

# -------------------- EDA --------------------
def perform_eda(df):
    st.header("📊 Exploratory Data Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Cities", df['city'].nunique())
    with col3:
        st.metric("Countries", df['country'].nunique())
    with col4:
        st.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")

    st.subheader("Pollutant Statistics")
    avg_pm25 = df['pm25_ugm3'].mean()
    max_pm25 = df['pm25_ugm3'].max()
    avg_no2 = df['no2_ugm3'].mean()
    max_no2 = df['no2_ugm3'].max()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Avg PM2.5 (μg/m³)", f"{avg_pm25:.2f}")
        st.metric("Max PM2.5 (μg/m³)", f"{max_pm25:.2f}")
    with c2:
        st.metric("Avg NO2 (μg/m³)", f"{avg_no2:.2f}")
        st.metric("Max NO2 (μg/m³)", f"{max_no2:.2f}")

    tab1, tab2, tab3 = st.tabs(["Time Series", "Distribution", "Geography"])
    with tab1:
        yearly_avg = df.groupby('year')['pm25_ugm3'].mean().reset_index()
        fig = px.line(yearly_avg, x='year', y='pm25_ugm3',
                      title='Average PM2.5 Concentration by Year',
                      labels={'pm25_ugm3': 'PM2.5 (μg/m³)', 'year': 'Year'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x='pm25_ugm3', nbins=50, title='PM2.5 Distribution')
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.box(df, x='season', y='pm25_ugm3', title='Seasonal PM2.5 Levels')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        fig = px.scatter_geo(sample_df, lat='latitude', lon='longitude',
                             color='pm25_ugm3', hover_name='city',
                             hover_data=['country', 'pm25_ugm3', 'no2_ugm3'],
                             title='Global PM2.5 Distribution')
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Model Training --------------------
def train_model(df):
    st.header("🤖 Predictive Modeling")
    st.info("Objective: Predict PM2.5 using location, time, and NO2")

    # Copy and check columns
    df_model = df.copy()
    required = ['year', 'month', 'no2_ugm3', 'latitude', 'longitude', 'country', 'city', 'season', 'pm25_ugm3']
    missing = ensure_columns(df_model, required)
    if missing:
        st.error(f"Missing columns for modeling: {missing}")
        return None, None, None

    # Optional sampling for speed
    use_sample = st.checkbox("Use sampling for faster training (recommended)", value=True)
    max_samples = st.number_input("Max sample size (if sampling)", min_value=1000, max_value=20000, value=5000, step=500)
    if use_sample and len(df_model) > max_samples:
        df_model = df_model.sample(max_samples, random_state=42)

    # Encoding: create mapping dicts for safe transform later
    # We'll map country and city to integers using pandas.factorize (stable and easy to get mapping)
    df_model['country_encoded'], country_uniques = pd.factorize(df_model['country'])
    df_model['city_encoded'], city_uniques = pd.factorize(df_model['city'])
    df_model['season_encoded'], season_uniques = pd.factorize(df_model['season'])

    # Save mapping dicts
    country_map = {k: v for v, k in enumerate(country_uniques)}
    city_map = {k: v for v, k in enumerate(city_uniques)}
    season_map = {k: v for v, k in enumerate(season_uniques)}

    feature_cols = ['year', 'month', 'no2_ugm3', 'latitude', 'longitude', 'country_encoded', 'season_encoded']
    X = df_model[feature_cols]
    y = df_model['pm25_ugm3']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection (smaller grid for speed)
    model_choice = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])
    if model_choice == "Random Forest":
        st.info("Training Random Forest (small grid for speed)")
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, None]
        }
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
    else:
        st.info("Training Gradient Boosting (small grid for speed)")
        gb = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        grid = GridSearchCV(gb, param_grid, cv=3, scoring='r2', n_jobs=-1)

    with st.spinner("Training model (this may take a moment)…"):
        grid.fit(X_train_scaled, y_train)

    model = grid.best_estimator_
    best_params = grid.best_params_

    # Predictions & metrics
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Performance")
        st.metric("R²", f"{r2_score(y_train, y_pred_train):.4f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f}")
        st.metric("MAE", f"{mean_absolute_error(y_train, y_pred_train):.2f}")
    with col2:
        st.subheader("Testing Performance")
        st.metric("R²", f"{r2_score(y_test, y_pred_test):.4f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")
        st.metric("MAE", f"{mean_absolute_error(y_test, y_pred_test):.2f}")

    st.subheader("Best Hyperparameters")
    st.json(best_params)

    # Feature importance (if available)
    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
        fig = px.bar(fi_df, x='importance', y='feature', orientation='h', title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Model does not expose feature_importances_. Skipping importance plot.")

    # Save model and artifacts
    os.makedirs('models', exist_ok=True)
    save_obj = {
        'model': model,
        'scaler': scaler,
        'mappings': {
            'country_map': country_map,
            'city_map': city_map,
            'season_map': season_map
        },
        'feature_cols': feature_cols
    }
    with open('models/air_quality_model.pkl', 'wb') as f:
        pickle.dump(save_obj, f)

    st.success("✅ Model and artifacts saved to models/air_quality_model.pkl")
    return model, scaler, save_obj['mappings'], feature_cols

# -------------------- Prediction Interface --------------------
def prediction_interface(original_df, model_artifact):
    st.header("🔮 Make Custom Predictions")
    if model_artifact is None:
        st.error("No trained model available. Please train a model first.")
        return

    model = model_artifact['model']
    scaler = model_artifact['scaler']
    mappings = model_artifact['mappings']
    feature_cols = model_artifact['feature_cols']

    # Input widgets
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.slider("Year", int(original_df['year'].min()), int(original_df['year'].max()), int(original_df['year'].median()))
        month = st.slider("Month", 1, 12, 6)
        no2 = st.number_input("NO2 (μg/m³)", min_value=0.0, max_value=500.0, value=30.0)
    with col2:
        country = st.selectbox("Country", sorted(original_df['country'].unique()))
        # cities filtered by chosen country
        cities = sorted(original_df[original_df['country'] == country]['city'].unique())
        city = st.selectbox("City", cities)
        latitude = st.number_input("Latitude", value=float(original_df[original_df['city'] == city]['latitude'].iloc[0]))
    with col3:
        longitude = st.number_input("Longitude", value=float(original_df[original_df['city'] == city]['longitude'].iloc[0]))
        season = st.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall'])

    # Safe encode using mappings; fallback to most frequent index if unseen
    def safe_map(mapping, key, fallback_map):
        if key in mapping:
            return mapping[key]
        # fallback to mode index
        if fallback_map:
            # fallback_map is a dict of value->index; pick the index for the first key (mode-ish)
            return list(fallback_map.values())[0]
        return 0

    country_enc = safe_map(mappings.get('country_map', {}), country, mappings.get('country_map'))
    city_enc = safe_map(mappings.get('city_map', {}), city, mappings.get('city_map'))
    season_enc = safe_map(mappings.get('season_map', {}), season, mappings.get('season_map'))

    if st.button("🎯 Predict PM2.5"):
        input_arr = np.array([[year, month, no2, latitude, longitude, country_enc, season_enc]])
        input_scaled = scaler.transform(input_arr)
        prediction = model.predict(input_scaled)[0]

        # Display
        st.subheader("Prediction Result")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted PM2.5", f"{prediction:.2f} μg/m³")
        with c2:
            if prediction < 12:
                cat = "Good"
                emoji = "🟢"
            elif prediction < 35.4:
                cat = "Moderate"
                emoji = "🟡"
            elif prediction < 55.4:
                cat = "Unhealthy for Sensitive"
                emoji = "🟠"
            else:
                cat = "Unhealthy"
                emoji = "🔴"
            st.metric("Air Quality", f"{cat} {emoji}")
        with c3:
            st.info(f"Location: {city}, {country}\nYear: {year}, Month: {month}\nNO2: {no2:.2f} μg/m³")

# -------------------- Main --------------------
def main():
    st.sidebar.title("🌍 Air Quality Analytics")
    st.sidebar.markdown("---")

    # Load data & metadata
    df = load_data()
    metadata = load_metadata()

    if df is None:
        st.error("Failed to load dataset. Please ensure 'data/air_quality_global.csv' exists.")
        return

    # Keep original for global sliders / lookups
    original_df = df.copy()

    # Preprocess with try/except to catch missing columns
    try:
        df = preprocess_data(df)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return

    # Year slider uses global range so it won't fail after country filtering
    year_min, year_max = int(original_df['year'].min()), int(original_df['year'].max())
    year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

    # Country filter
    countries = ['All'] + sorted(original_df['country'].dropna().unique().tolist())
    selected_country = st.sidebar.selectbox("Country", countries)

    # Apply filters in a safe order (year range first then country)
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    if selected_country != 'All':
        df = df[df['country'] == selected_country]

    # Navigation
    page = st.sidebar.radio("Navigation", ["📖 About & Metadata", "🔍 Data Explorer", "📊 EDA", "🤖 Modeling", "🔮 Predictions"])
    st.sidebar.markdown("---")

    # Auto-load saved model into session_state (if present)
    if 'model_artifact' not in st.session_state:
        model_path = 'models/air_quality_model.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                    st.session_state['model_artifact'] = saved
                    st.success("Loaded saved model from models/air_quality_model.pkl")
            except Exception:
                st.warning("Saved model found but failed to load. You can retrain.")

    if page == "📖 About & Metadata":
        st.title("📖 About This Application")
        st.markdown("Urban Air Quality Analysis & Prediction focusing on PM2.5 and NO2.")

        if metadata:
            st.subheader("📋 Dataset Metadata")
            st.json({
                'dataset_name': safe_get(metadata, 'dataset_name'),
                'version': safe_get(metadata, 'version'),
                'creation_date': safe_get(metadata, 'creation_date'),
                'total_records': safe_get(metadata, 'total_records'),
                'license': safe_get(metadata, 'license')
            })

            if 'data_quality_notes' in metadata:
                st.subheader("⚠️ Data Quality Notes")
                st.warning(safe_get(metadata, 'data_quality_notes', 'No notes provided.'))

            if 'usage_recommendations' in metadata:
                st.subheader("💡 Usage Recommendations")
                st.info(safe_get(metadata, 'usage_recommendations', 'None provided.'))
        else:
            st.info("No metadata.json found or failed to parse. Make sure data/metadata.json is present.")

    elif page == "🔍 Data Explorer":
        st.title("🔍 Data Explorer")
        st.dataframe(df.head(200), use_container_width=True)
        st.subheader("Dataset Summary")
        st.write(df.describe(include='all'))
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")

    elif page == "📊 EDA":
        perform_eda(df)

    elif page == "🤖 Modeling":
        model, scaler, mappings, feature_cols = train_model(df)
        if model is not None:
            # Save and set session_state (saved inside train_model already)
            model_path = 'models/air_quality_model.pkl'
            try:
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                    st.session_state['model_artifact'] = saved
            except Exception:
                st.warning("Model trained but failed to reload saved artifact into session_state.")

    elif page == "🔮 Predictions":
        if 'model_artifact' not in st.session_state:
            st.warning("Please train a model first in the 'Modeling' section or ensure a saved model exists.")
        else:
            prediction_interface(original_df, st.session_state['model_artifact'])

if __name__ == "__main__":
    main()
