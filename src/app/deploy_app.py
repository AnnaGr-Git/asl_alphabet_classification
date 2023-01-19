import subprocess
from pathlib import Path
import os.path
from fastapi import FastAPI, File, Response, UploadFile

app = FastAPI()


@app.get("/")
def read_root() -> dict[str, str]:
    """Hwllo world function"""
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int) -> dict[str, int]:
    """Function to read the items"""
    return {"item_id": item_id}


@app.get("/train")
def train() -> Response:
    """Function to train the model"""
    result = subprocess.run(
        ["python", "-u", "src/models/train_model.py"], check=True, capture_output=True
    )
    return Response(content=result.stdout, media_type="text/plain")


@app.post("/predict")
async def predict(data: UploadFile = File(...)) -> Response:
    """Function to predict the asl letter given an image"""
    print("We are in the predict function")
    # Read uploaded file to image
    img_path = "image.jpg"
    with open(img_path, "wb") as image:
        content = await data.read()
        print("In the with open")
        image.write(content)
        image.close()

    path = Path(img_path)
    print(path.is_file())
    print(path.exists())

    result = subprocess.run(
        ["python", "-u", "src/models/predict_model.py", img_path], check=True, capture_output=True
    )
    return Response(content=result.stdout, media_type="text/plain")

