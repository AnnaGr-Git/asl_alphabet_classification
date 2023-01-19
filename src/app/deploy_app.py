import subprocess

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
# async def predict(data: Union[UploadFile, str] = File(...)):
async def predict(data: UploadFile = File(...)) -> Response:
    """Function to predict the asl letter given an image"""
    # Read uploaded file to image
    img_path = "image.jpg"
    with open(img_path, "wb") as image:
        content = await data.read()
        image.write(content)
        image.close()

    result = subprocess.run(
        ["python", "-u", "src/models/predict_model.py", img_path], check=True, capture_output=True
    )
    return Response(content=result.stdout, media_type="text/plain")


# async def predict(data: str):
#     # Read uploaded file to image
#     if type(data) == UploadFile:
#         img_path = 'image.jpg'
#         with open(img_path, 'wb') as image:
#             content = await data.read()
#             image.write(content)
#             image.close()
#     else:
#         img_path = data

#     #img = Image.open("image.jpg")
#     result = subprocess.run(['python', '-u', 'src/models/predict_model.py', img_path],
#                               check=True, capture_output=True)
#     return Response(content=result.stdout, media_type="text/plain")
