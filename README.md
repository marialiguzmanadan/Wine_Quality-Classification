
# Wine Quality Prediction Model Deployment

## Overview

This repository contains all the necessary files for deploying a machine learning model that predicts the quality of wine based on its chemical properties. The model is pre-trained and ready to use with a Flask-based web service for easy integration into applications.

## Directory Structure

```plaintext
Wine Quality/
│
├── models/
│   ├── best_model.pkl           # Pre-trained machine learning model
│   └── scaler.pkl               # Pre-trained scaler for input normalization
│
├── input.csv                    # Example CSV file with input data
├── model_performance.txt        # Model performance metrics or evaluation results
├── app.py                       # Flask service to serve predictions
├── Wine_Quality.ipynb           # Model training and evaluation
└── wine.py                      # Handles model loading, preprocessing, and predictions.
```

## Getting Started

### Prerequisites

Ensure you have Python installed, along with the following packages:
- Flask
- joblib
- pandas
- numpy
- scikit-learn

You can install these packages using pip:

```bash
pip install flask joblib pandas numpy scikit-learn
```

### Usage

#### Model and Scaler

- **`best_model.pkl`**: Serialized trained model for wine quality prediction.
- **`scaler.pkl`**: StandardScaler instance used to normalize the input features.

#### Input Data

- **`input.csv`**: Template for formatting new input data for predictions. Make sure your input data matches the model's expected feature set.

#### Flask Service

- **`app.py`**: Script to run a Flask web server that handles prediction requests.
  - **Run the service** with:
    ```bash
    python app.py
    ```
  - **Making Predictions**:
    Send a POST request to `http://localhost:5000/predict` with JSON payload containing the input features. The service will return the predicted wine quality.

#### Model Training and Evaluation

- **`wine.py`**: Script containing the model's training and evaluation logic, used for further development or retraining.

### Example POST Request using cURL

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"fixed_acidity": 7.4, "volatile_acidity": 0.7, "chlorides": 0.076, "free_sulfur_dioxide": 11, "total_sulfur_dioxide": 34, "density": 0.9978, "alcohol": 9.4}'
```

## Conclusion

This setup provides a complete framework for deploying a machine learning model to predict wine quality in a production environment. By following this guide, users can effectively manage and utilize the model for testing and real-world applications.
