from flask import Flask, request, jsonify
#import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('results/solar-lstm-model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    forecast = model.predict(df)
    return jsonify(forecast[['ds', 'yhat']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
