from flask import Flask,jsonify,request
import pickle
import joblib
import numpy as np  
import pandas as pd

app=Flask(__name__)

model=joblib.load('model.joblib')

@app.route('/')
def home():
    return 'Hello World'


@app.route('/predict',methods=['POST'])
def predict():
    # data = request.get_json()
   
    intercept = 1
    age = request.form.get('Age')
    dribbling = request.form.get('Dribbling / Reflexes')
    passing =  request.form.get('Passing / Kicking')
    shooting = request.form.get('Shooting / Handling')
    reputation = request.form.get('International reputation')
    mentality = request.form.get('Total mentality')
    shot_power = request.form.get('Shot power')
    power = request.form.get('Total power')
    ball_control = request.form.get('Ball control')
    finishing = request.form.get('Finishing')


    received_data = [{
        'Intercept':1, 
        'Age': age,
        'Dribbling / Reflexes': dribbling,
        'Passing / Kicking': passing,
        'Shooting / Handling': shooting,
        'International reputation': reputation,
        'Total mentality': mentality,
        'Shot power': shot_power,
        'Total power': power,
        'Ball control': ball_control,
        'Finishing': finishing
    }]

     # Return the received data as JSON response
    input_data = pd.DataFrame(received_data)
    input_array = input_data.values
    prediction = model.predict(input_array)
    predicted_values_original_scale = np.exp(prediction)
    formatted_prediction = "{:,.2f}".format(predicted_values_original_scale[0])

    # return jsonify({'prediction': prediction[0]})
    # return jsonify({'prediction': predicted_values_original_scale[0]})    
    # "prediction": 119009354.24391253  want 2 values after decimal
    return jsonify({'prediction': formatted_prediction} )



if __name__=='__main__':
    app.run(debug=True)

