from fastapi import FastAPI, Response
import subprocess

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.get("/train")
def train():
    result = subprocess.run(['python', '-u', 'src/models/train_model.py'], check=True, capture_output=True)
    return Response(content=result.stdout, media_type="text/plain")

@app.get("/predict/{i}")
def predict(i: int):
    result = subprocess.run(['python', '-u', 'src/models/predict_model.py', str(i)], check=True, capture_output=True)
    return Response(content=result.stdout, media_type="text/plain")
    