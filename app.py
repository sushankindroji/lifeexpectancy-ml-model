import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@flask_app.route("/")
def about():
    return render_template("about.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Define the columns
    columns = [
        'Country', 'Region', 'Year', 'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 
        'Alcohol_consumption', 'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 
        'Incidents_HIV', 'GDP_per_capita', 'Population_mln', 'Thinness_ten_nineteen_years', 
        'Schooling', 'Economy_status', 'Life_expectancy'
    ]
    
    # Extract input data from the form
    input_data = [request.form[column] for column in columns[:-1]]  # Exclude 'Life_expectancy'
    
    # Select the relevant 8 features for prediction
    selected_features = [
        'Year', 'Under_five_deaths', 'Adult_mortality', 'BMI', 
        'Diphtheria', 'Polio', 'Thinness_ten_nineteen_years', 'Schooling'
    ]
    
    # Create a dictionary from the input data
    input_dict = dict(zip(columns[:-1], input_data))
    
    # Extract the selected features' values
    selected_data = [float(input_dict[feature]) for feature in selected_features]
    
    # Scale the selected data
    features = scaler.transform([np.array(selected_data)])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    return render_template("result.html", prediction_text="{:.2f}".format(prediction))

@flask_app.route("/predict_page")
def predict_page():
    return render_template("index.html")

if __name__ == "__main__":
    flask_app.run(debug=True)
