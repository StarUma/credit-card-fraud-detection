from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.src.prediction_service import make_prediction

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    try:
        features = [
            data.get("scaled_time", 0.0),
            data.get("scaled_amount", 0.0)
        ]

        # Add V1 to V28, use 0.0 if not provided
        for i in range(1, 29):
            features.append(data.get(f"V{i}", 0.0))

        prediction = make_prediction(features)
        return {"prediction": prediction}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
