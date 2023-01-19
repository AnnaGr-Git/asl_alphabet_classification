from fastapi import FastAPI, Response, UploadFile, File
import subprocess
from PIL import Image
from typing import Union
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

@app.post("/predict")
#async def predict(data: Union[UploadFile, str] = File(...)):
async def predict(data: UploadFile = File(...)):
    # Read uploaded file to image
    img_path = 'image.jpg'
    with open(img_path, 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
    
    result = subprocess.run(['python', '-u', 'src/models/predict_model.py', img_path], check=True, capture_output=True)
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
#     result = subprocess.run(['python', '-u', 'src/models/predict_model.py', img_path], check=True, capture_output=True)
#     return Response(content=result.stdout, media_type="text/plain")
    