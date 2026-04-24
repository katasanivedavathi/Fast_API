# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# import pickle
# import numpy as np

# app = FastAPI(title="Student Mark Predictor")

# # Mount static files
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Templates
# templates = Jinja2Templates(directory="templates")

# # Load model
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)


# @app.get("/", response_class=HTMLResponse)
# def home(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/predict", response_class=HTMLResponse)
# def predict(
#     request: Request,
#     study_hours: float = Form(...),
#     previous_score: float = Form(...)
# ):
#     input_data = np.array([[study_hours, previous_score]])
#     prediction = model.predict(input_data)[0]

#     return templates.TemplateResponse(
#         "result.html",
#         {
#             "request": request,
#             "prediction": round(prediction, 2)
#         }
#     )




from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load model once
model = pickle.load(open("model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "FastAPI is working"}

@app.get("/predict")
def predict(hours: float):
    # convert to DataFrame
    input_data = pd.DataFrame([[hours]], columns=["Study Hours"])
    result = model.predict(input_data)
    return {"Predicted Score": float(result[0])}
