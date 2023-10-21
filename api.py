from json import load
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
#from PIL import Image
from io import BytesIO
from numpy import array as nparray

from fastapi import FastAPI, UploadFile


MODEL_DIR = 'model'
MODEL = load_model(MODEL_DIR)
TARGET_SIZE = (244, 244)
with open('classes.json') as f:
    CLASSES = load(f)

def load_image(image_io):
    return nparray([
        img_to_array(
            load_img(
                image_io,
                #target_size=TARGET_SIZE,
                #interpolation='bilinear'
            )
        )
    ])


app = FastAPI()

@app.post("/predict-image")
async def predict_image(file: UploadFile):
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

