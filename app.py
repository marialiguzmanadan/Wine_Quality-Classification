from flask import Flask, request, jsonify
import pandas as pd
import wine

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_api():
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        input_data = pd.DataFrame([data])
        predicted_quality = wine.predict(input_data)
        return jsonify({'predicted_quality': int(predicted_quality)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
