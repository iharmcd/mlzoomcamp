import pickle
import numpy as np

from flask import Flask, request, jsonify


with open('model2.bin', 'rb') as f_model:
    model = pickle.load(f_model)

with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


app = Flask('subscription')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'subicription_probability': float(prediction),
        'subscription': bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)