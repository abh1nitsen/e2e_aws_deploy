from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Looad the pkl files we saved earlier
with open("12510050_logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("12510050_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the numeric columns
numericCols = [
    'age', 'campaign', 'pDays', 'previous',
    'consPriceIdx', 'consConfIdx', 'eurIbor3M',
    'nrEmployed', 'pDaysFlag'
]

# Expected feature order for prediction
feature_order = ['age',
 'campaign',
 'pDays',
 'previous',
 'empVarRate',
 'consPriceIdx',
 'consConfIdx',
 'eurIbor3M',
 'nrEmployed',
 'pDaysFlag',
 'job_blue-collar',
 'job_entrepreneur',
 'job_housemaid',
 'job_management',
 'job_retired',
 'job_self-employed',
 'job_services',
 'job_student',
 'job_technician',
 'job_unemployed',
 'job_unknown',
 'mStatus_married',
 'mStatus_single',
 'mStatus_unknown',
 'education_basic.6y',
 'education_basic.9y',
 'education_high.school',
 'education_illiterate',
 'education_professional.course',
 'education_university.degree',
 'education_unknown',
 'default_unknown',
 'default_yes',
 'housing_unknown',
 'housing_yes',
 'loan_unknown',
 'loan_yes',
 'contact_telephone',
 'month_aug',
 'month_dec',
 'month_jul',
 'month_jun',
 'month_mar',
 'month_may',
 'month_nov',
 'month_oct',
 'month_sep',
 'dayOfWeek_mon',
 'dayOfWeek_thu',
 'dayOfWeek_tue',
 'dayOfWeek_wed',
 'pastOutcome_nonexistent',
 'pastOutcome_success'] 


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON input 
        input_data = request.json
        
        # Convert input to dataframe
        df_input = pd.DataFrame([input_data])

        # Ensure all columns exist
        for col in feature_order:
            if col not in df_input.columns:
                df_input[col] = 0  # default for missing dummy variables
        
        # Reorder columns to match training
        df_input = df_input[feature_order]

        # Scale numeric columns
        df_input[numericCols] = scaler.transform(df_input[numericCols])
        
        # Make prediction
        pred = model.predict(df_input)[0]

        return jsonify({"prediction": int(pred)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is working. Use /predict to submit data."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)