from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    form_values = request.form.to_dict()
    float_fields = ['Fare']
    int_fields = ['Age','Pclass_2','Pclass_3','Sex_male','Pclass_2','Embarked_Q','Embarked_S','family_type_nuclear','family_type_big']

    features=[]
    for key,values in form_values.items():
        if key in float_fields:
            features.append(float(values))
        elif key in int_fields:
            features.append(int(values))
        else:
            features.append(values)

    # int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Survived' if prediction[0] == 1 else 'Not Survived'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True,port=8080)