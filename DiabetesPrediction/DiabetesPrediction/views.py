from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Load dataset
    data = pd.read_csv(r'E:\ai_project\DiabetesPrediction\DiabetesPrediction\Bangladesh.csv')

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['gender'] = label_encoder.fit_transform(data['gender'])  # Male=1, Female=0
    data['family_diabetes'] = label_encoder.fit_transform(data['family_diabetes'])
    data['hypertensive'] = label_encoder.fit_transform(data['hypertensive'])
    data['family_hypertension'] = label_encoder.fit_transform(data['family_hypertension'])
    data['cardiovascular_disease'] = label_encoder.fit_transform(data['cardiovascular_disease'])
    data['stroke'] = label_encoder.fit_transform(data['stroke'])

    # Split dataset
    X = data.drop("diabetic", axis=1)  # Assuming "diabetic" is the target column
    Y = data["diabetic"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, Y_train)

    # Get values from user input
    age = float(request.GET['age'])
    gender = request.GET['gender']  # Male or Female
    pulse_rate = float(request.GET['pulse_rate'])
    systolic_bp = float(request.GET['systolic_bp'])
    diastolic_bp = float(request.GET['diastolic_bp'])
    glucose = float(request.GET['glucose'])
    height = float(request.GET['height'])
    weight = float(request.GET['weight'])
    bmi = float(request.GET['bmi'])
    family_diabetes = request.GET['family_diabetes']
    hypertensive = request.GET['hypertensive']
    family_hypertension = request.GET['family_hypertension']
    cardiovascular_disease = request.GET['cardiovascular_disease']
    stroke = request.GET['stroke']

    # Convert categorical inputs to numerical values
    gender = 1 if gender.lower() == "male" else 0
    family_diabetes = 1 if family_diabetes.lower() == "yes" else 0
    hypertensive = 1 if hypertensive.lower() == "yes" else 0
    family_hypertension = 1 if family_hypertension.lower() == "yes" else 0
    cardiovascular_disease = 1 if cardiovascular_disease.lower() == "yes" else 0
    stroke = 1 if stroke.lower() == "yes" else 0

    # Create input array for prediction
    input_features = [[age, gender, pulse_rate, systolic_bp, diastolic_bp, glucose, height, weight, bmi,
                       family_diabetes, hypertensive, family_hypertension, cardiovascular_disease, stroke]]

    # Make prediction
    pred = model.predict(input_features)

    # Interpret the prediction
    result_text = "Positive" if pred == [1] else "Negative"

    return render(request, 'predict.html', {"result2": result_text})
