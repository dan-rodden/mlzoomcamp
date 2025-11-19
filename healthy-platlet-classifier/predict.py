import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

model_file="xgb_model_eta=0.1_depth=3_min-child=6_v1.0.bin"

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# name of the flask app
app = Flask('healthy_blood_platlet_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    # get the samples
    samples = request.get_json()

    # transform the samples (sparse matrix)
    X = dv.transform([samples])
    # convert to DMatrix 
    dX = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    # predict
    y_pred = model.predict(dX)
    prediction = y_pred[0]

    result = {
        'probability': float(prediction),
        'is_healthy': bool(prediction >= 0.5) # 1 is healthy
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)