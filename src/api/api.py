from json import load
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from io import BytesIO
from numpy import array as nparray

from fastapi import FastAPI, UploadFile, HTTPException


MODEL_DIR = 'model'
MODEL = load_model(MODEL_DIR)
TARGET_SIZE = MODEL.input_shape[1:3]
with open('classes.json') as f:
    CLASSES = load(f)

def load_image(image_io):
    return nparray([
        img_to_array(
            load_img(
                image_io,
                target_size=TARGET_SIZE,
                interpolation='bilinear',
                keep_aspect_ratio=True
            )
        )
    ])


app = FastAPI()

@app.post("/predict-image")
async def predict_image(file: UploadFile):

    allowed_content_types = ["image/jpeg", "image/png"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(status_code=415, detail="Unsupported file format")

    max_file_size = 1024 * 1024  # 1 MiB in bytes
    if file.file.tell() > max_file_size:
        raise HTTPException(status_code=413, detail="File size exceeds 1 MiB")

    file_bytes = await file.read()
    img = load_image(BytesIO(file_bytes))
    probs = MODEL.predict(img)[0]
    arg3 = probs.argsort()[-3:][::-1]
    return {
        f'pred_{i}': (CLASSES[arg], float(probs[arg]))
        for i, arg in enumerate(arg3)
    }


@app.get('/')
async def home():
    return {'message': 'Hello World'}

