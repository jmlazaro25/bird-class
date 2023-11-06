from requests import post as requests_post
import os

from requests.models import Response
from streamlit.runtime.uploaded_file_manager import UploadedFile


BMAPI_URL = os.environ['BMAPI_URL']

def request_prediction(image_file: UploadedFile) -> Response:
    return requests_post(
        f'{BMAPI_URL}/predict-image',
        files={'file': (image_file.name, image_file.read(), image_file.type)}
    )

def data_from_json(json: dict) -> dict:
    return {
        species.title(): prob for species, prob in json.values()
    }

