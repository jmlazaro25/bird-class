from requests import post as requests_post

from requests.models import Response
from streamlit.runtime.uploaded_file_manager import UploadedFile


def request_prediction(image_file: UploadedFile) -> Response:
    return requests_post(
        'http://localhost:8000/predict-image',
        files={'file': (image_file.name, image_file.read(), image_file.type)}
    )

def data_from_json(json: dict) -> dict:
    return {
        species.title(): prob for species, prob in json.values()
    }

