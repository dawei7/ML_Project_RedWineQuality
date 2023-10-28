from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the saved model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the saved scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_form(request: Request):
    return templates.TemplateResponse("predict_wine_quality_form.html", {"request": request})


@app.post("/predict")
async def make_prediction(request: Request):
    form_data = await request.form()
    features = np.array([[
        float(form_data['fixed_acidity']),
        float(form_data['volatile_acidity']),
        float(form_data['citric_acid']),
        float(form_data['residual_sugar']),
        float(form_data['chlorides']),
        float(form_data['free_sulfur_dioxide']),
        float(form_data['total_sulfur_dioxide']),
        float(form_data['density']),
        float(form_data['pH']),
        float(form_data['sulphates']),
        float(form_data['alcohol'])
    ]])
    # Normalize the input data
    normalized_features = scaler.transform(features)
    prediction = model.predict(normalized_features)
    prediction_probabilities = model.predict_proba(normalized_features)
    result = "Excellent" if prediction[0] == 1 else "Not Excellent"
    excellent_prob = round(prediction_probabilities[0][1], 2)
    not_excellent_prob = round(prediction_probabilities[0][0], 2)

    return templates.TemplateResponse("prediction_results.html", {
        "request": request,
        "result": result,
        "excellent_prob": excellent_prob,
        "not_excellent_prob": not_excellent_prob
    })


# uvicorn predict:app --reload --port 8080
# http://localhost:8080/
