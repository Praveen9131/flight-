import streamlit as st
import pandas as pd
import numpy as np
import pickle

def predict_flight_delay(sample_input, model_path='/major/flight_delay_model1.pkl'):
    """Predict flight delay using the pickle file and sample input, or apply if-else rule."""
    # If-else rule for extreme conditions
    if (sample_input['wind_speed'] > 50.0 and 
        sample_input['pressure'] < 980.0 and 
        sample_input['rain_1h'] > 30.0):
        if st.session_state.get('debug', False):
            st.write("Prediction Source: If-Else Rule (Extreme wind, pressure, and rain)")
        return "Delayed"
    
    # Model-based prediction
    try:
        # Load pickle file
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # Validate pickle contents
        required_keys = {'model', 'scaler', 'imputer', 'features'}
        if not all(key in saved_data for key in required_keys):
            raise KeyError(f"Pickle file missing required keys: {required_keys - set(saved_data.keys())}")
        
        model = saved_data['model']
        scaler = saved_data['scaler']
        imputer = saved_data['imputer']
        expected_features = saved_data['features']
        
        # Prepare sample input
        input_data = pd.DataFrame([sample_input])
        
        # Define numeric features to match imputer expectations
        numeric_features = [
            'temp', 'visibility', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 
            'pressure', 'humidity', 'wind_speed', 'wind_deg', 'wind_gust', 'rain_1h', 
            'rain_3h', 'clouds_all'
        ]
        
        # Validate input
        missing_features = [f for f in numeric_features if f not in sample_input]
        if missing_features:
            st.warning(f"Sample input missing features: {missing_features}")
        
        # Impute numeric features
        input_data_imputed = pd.DataFrame(
            imputer.transform(input_data[numeric_features]),
            columns=numeric_features
        )
        
        # Drop correlated features and feels_like (not in expected_features)
        corr_dropped = ['dew_point', 'temp_max', 'temp_min', 'feels_like']
        input_data_imputed = input_data_imputed.drop(columns=[f for f in corr_dropped if f in input_data_imputed.columns])
        
        # Add weather dummies
        weather_features = [f for f in expected_features if f.startswith('weather_')]
        for feature in expected_features:
            if feature in weather_features:
                input_data_imputed[feature] = sample_input.get(feature, 0)
            elif feature not in input_data_imputed.columns:
                input_data_imputed[feature] = 0
        
        # Reorder columns to match expected features
        input_data_imputed = input_data_imputed[expected_features]
        
        # Scale
        input_data_scaled = scaler.transform(input_data_imputed)
        
        # Predict
        prediction = model.predict(input_data_scaled)
        
        # Debug: Show raw prediction, scaled features, and scaler parameters if enabled
        if st.session_state.get('debug', False):
            st.write("Prediction Source: SVC Model")
            st.write(f"Raw Prediction Value: {prediction[0]}")
            st.write(f"Scaled Input Features: {input_data_scaled[0]}")
            st.write(f"Scaler Means: {scaler.mean_}")
            st.write(f"Scaler Std Dev: {np.sqrt(scaler.var_)}")
        
        return "Delayed" if prediction[0] == 1 else "On-time"
    
    except FileNotFoundError:
        st.error(f"Pickle file not found at {model_path}.")
        return None
    except KeyError as e:
        st.error(f"Missing key in pickle file: {e}")
        return None
    except ValueError as e:
        st.error(f"Input preprocessing failed: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# Streamlit app
st.title("Flight Delay Prediction")
st.write("Enter weather data to predict if a flight is on-time or delayed.")

# Input form for numeric features
st.subheader("Weather Input")
col1, col2 = st.columns(2)

numeric_inputs = {
    'temp': col1.number_input("Temperature (°F)", value=35.0, step=0.1),
    'visibility': col1.number_input("Visibility (meters)", value=100.0, step=10.0),
    'dew_point': col1.number_input("Dew Point (°F)", value=33.0, step=0.1),
    'feels_like': col1.number_input("Feels Like (°F)", value=30.0, step=0.1),
    'temp_min': col1.number_input("Min Temperature (°F)", value=33.0, step=0.1),
    'temp_max': col1.number_input("Max Temperature (°F)", value=37.0, step=0.1),
    'pressure': col1.number_input("Pressure (hPa)", value=970.0, step=1.0),
    'humidity': col2.number_input("Humidity (%)", value=100.0, step=1.0),
    'wind_speed': col2.number_input("Wind Speed (mph)", value=60.0, step=1.0),
    'wind_deg': col2.number_input("Wind Direction (degrees)", value=270.0, step=1.0),
    'wind_gust': col2.number_input("Wind Gust (mph)", value=80.0, step=1.0),
    'rain_1h': col2.number_input("Rainfall (1h, mm)", value=40.0, step=1.0),
    'rain_3h': col2.number_input("Rainfall (3h, mm)", value=100.0, step=1.0),
    'clouds_all': col2.number_input("Cloud Cover (%)", value=100.0, step=1.0)
}

# Weather condition dropdown
st.subheader("Weather Condition")
weather_conditions = [
    'weather_201', 'weather_211', 'weather_300', 'weather_301', 'weather_500', 
    'weather_501', 'weather_502', 'weather_503', 'weather_600', 'weather_601', 
    'weather_615', 'weather_701', 'weather_711', 'weather_721', 'weather_741', 
    'weather_800', 'weather_801', 'weather_802', 'weather_803', 'weather_804'
]
selected_weather = st.selectbox("Select Weather Condition", weather_conditions, index=weather_conditions.index('weather_211'))
weather_inputs = {condition: 1 if condition == selected_weather else 0 for condition in weather_conditions}

# Combine inputs
sample_input = {**numeric_inputs, **weather_inputs}

# Debug toggle
st.checkbox("Enable Debug Output (shows prediction source, raw prediction, scaled features, and scaler parameters)", key="debug")

# Predict button
if st.button("Predict"):
    prediction = predict_flight_delay(sample_input)
    if prediction:
        st.subheader("Prediction")
        st.write(f"**{prediction}**")

# Test all weather conditions
if st.button("Test All Weather Conditions"):
    results = []
    for condition in weather_conditions:
        weather_inputs = {c: 1 if c == condition else 0 for c in weather_conditions}
        sample_input = {**numeric_inputs, **weather_inputs}
        prediction = predict_flight_delay(sample_input)
        results.append(f"{condition}: {prediction}")
    st.subheader("Results for All Weather Conditions")
    for result in results:
        st.write(result)
