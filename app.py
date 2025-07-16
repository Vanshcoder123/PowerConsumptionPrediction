# from flask import Flask, render_template, request
# import joblib
# import numpy as np

# app = Flask(__name__)

# # Load model and encoders
# model = joblib.load("energy_model.pkl")
# label_encoders = joblib.load("label_encoders.pkl")

# @app.route('/')
# def index():
#     return render_template("form.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         building_type = request.form['building_type']
#         square_footage = int(request.form['square_footage'])
#         occupants = int(request.form['occupants'])
#         appliances = int(request.form['appliances'])
#         temperature = float(request.form['temperature'])
#         day_of_week = request.form['day_of_week']

#         # Encode categorical values
#         building_encoded = label_encoders['Building Type'].transform([building_type])[0]
#         day_encoded = label_encoders['Day of Week'].transform([day_of_week])[0]

#         # Combine into array
#         input_data = np.array([[building_encoded, square_footage, occupants, appliances, temperature, day_encoded]])

#         # Predict
#         prediction = model.predict(input_data)[0]

#         return f"<h2>✅ Predicted Energy Consumption: {prediction:.2f} kWh</h2>"

#     except Exception as e:
#         return f"<h3>Error: {str(e)}</h3>"

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoders
model = joblib.load("energy_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

@app.route('/')
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        building_type = request.form['building_type']
        square_footage = int(request.form['square_footage'])
        occupants = int(request.form['occupants'])
        appliances = int(request.form['appliances'])
        temperature = float(request.form['temperature'])
        day_of_week = request.form['day_of_week']

        # Encode categorical inputs
        building_encoded = label_encoders['Building Type'].transform([building_type])[0]
        day_encoded = label_encoders['Day of Week'].transform([day_of_week])[0]

        # Prepare input for model
        input_data = np.array([[building_encoded, square_footage, occupants, appliances, temperature, day_encoded]])

        # Predict using the model
        prediction = model.predict(input_data)[0]

        # Render result page
        return render_template("result.html", prediction=prediction)

    except Exception as e:
        return f"<h3 style='color:red;'>❌ Error: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
