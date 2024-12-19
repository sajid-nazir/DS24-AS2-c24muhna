import streamlit as st
import mlflow
import mlflow.tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Load the model
model_uri = "runs:/157958cbd0e64c819a98fc9693844cd8/model"
model = mlflow.tensorflow.load_model(model_uri)

# Define the input features
categorical_features = ['month', 'day']
numerical_features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']
input_features = numerical_features + categorical_features

# Define options for categorical features
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_options = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

# Streamlit app
st.title("Forest Fire Area Prediction")

# Creating input fields for numerical features
input_data = {}
for feature in numerical_features:
    input_data[feature] = st.text_input(f"Enter {feature}")

# Creating dropdown menus for categorical features
input_data['month'] = st.selectbox("Select month", month_options)
input_data['day'] = st.selectbox("Select day", day_options)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess input data
if st.button("Predict"):
    # Handle missing values in case user doesn't input all the input values on streamlit 
    imputer = SimpleImputer(strategy='most_frequent')
    input_df[categorical_features] = imputer.fit_transform(input_df[categorical_features])
    input_df[numerical_features] = imputer.fit_transform(input_df[numerical_features])

    # Encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[month_options, day_options])
    encoded_categorical = encoder.fit_transform(input_df[categorical_features])

    # Scale numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(input_df[numerical_features])

    # Combine the processed features
    processed_data = np.hstack((scaled_numerical, encoded_categorical))

    # Check the shape of the processed data
    #st.write(f"Processed data shape: {processed_data.shape}")

    # Make sure the processed data has the correct shape, faced this issue for quite a lot of time
    if processed_data.shape[1] == 27:
        # Make prediction
        prediction = model.predict(processed_data)
        # Use expm1 to reverse log1p transformation because we had applied log transformation on our area column before training the model
        st.write(f"Predicted Area: {np.expm1(prediction[0][0])}")  
    else:
        st.write(f"Expected 27 features, but got {processed_data.shape[1]} features.")
