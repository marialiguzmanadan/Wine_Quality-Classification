import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model = joblib.load('models/best_model.pkl')

# Load the scaler used during training
scaler = joblib.load('models/scaler.pkl')

# Define the selected features
selected_features = ['fixed_acidity', 'volatile_acidity', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'alcohol']

# Define the full set of features expected by the scaler
all_features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 
                'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

def preprocess_input(input_data):
    """
    Preprocess the input data using StandardScaler.
    
    :param input_data: DataFrame with the selected features.
    :return: scaled input data.
    """
    # Create a DataFrame with all features initialized to zero
    input_df = pd.DataFrame([np.zeros(len(all_features))], columns=all_features)
    
    # Update the DataFrame with the actual input values for the selected features
    for feature in selected_features:
        input_df[feature] = input_data[feature]
    
    # Apply the scaler to all features
    input_scaled = scaler.transform(input_df)
    
    # Convert scaled input to a DataFrame and select only the features used by the model
    input_scaled_df = pd.DataFrame(input_scaled, columns=all_features)
    input_final = input_scaled_df[selected_features]
    
    return input_final

def predict(input_data):
    """
    Make a prediction based on the input data using the trained model.
    
    :param input_data: DataFrame with the selected features.
    :return: predicted quality (classification label).
    """
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)
    
    # Make prediction using the model
    prediction = model.predict(preprocessed_data)
    
    return prediction[0]

if __name__ == "__main__":
    # Load the input data from a CSV file
    input_csv = 'input.csv'  # Path to the input CSV file
    input_data = pd.read_csv(input_csv)
    
    # Ensure that the CSV contains only one row of data
    if input_data.shape[0] != 1:
        raise ValueError("The input CSV should contain exactly one row of data.")
    
    # Extract the relevant features
    input_example = input_data[selected_features].iloc[0]
    
    # Call predict function
    predicted_quality = predict(input_example)
    
    # Output the prediction
    print(f"Predicted Wine Quality: {predicted_quality}")
