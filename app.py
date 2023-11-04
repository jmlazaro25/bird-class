import streamlit as st

from requests import post as requests_post
from pandas import DataFrame
import altair as alt

# For typing
from requests.models import Response
from streamlit.runtime.uploaded_file_manager import UploadedFile


CLASS_STR = 'Species'
PROB_STR = 'Probability'


def request_prediction(image_file: UploadedFile) -> Response:
    return requests_post(
        'http://localhost:8000/predict-image',
        files={'file': (image_file.name, image_file.read(), image_file.type)}
    )

def data_from_json(json: dict) -> dict:
    data = {
        species.title(): prob for species, prob in json.values()
    }
    return data

def plot_species_prob(data: dict) -> None:
    data_df = DataFrame({
        CLASS_STR: [species for species, _ in data.items()],
        PROB_STR: [round(prob * 100, 1) for _, prob in data.items()]
    })
    chart = alt.Chart(data_df).mark_bar().encode(
        x=alt.X(
            f'{CLASS_STR}:O',
            sort=None,
            axis=alt.Axis(labelAngle=-45, labelLimit=150)
        ),
        y=alt.Y(f'{PROB_STR}:Q', title=f'{PROB_STR} (%)')
    )
    chart = chart.configure_text(
        fontSize=50,
        fontWeight='bold'
    )
    st.altair_chart(chart, use_container_width=True)

def main():

    st.set_page_config(
        page_title='bird-class',
        page_icon=':penguin:'
    )

    st.title(':penguin: Bird Class :student:')

    st.write('Upload an image below to predict a classification for your bird image.')
    image_file = st.file_uploader(
        'Upload bird image $\leq$ 1 MiB',
        type=['.png', 'jpeg'],
    )

    # Don't do anything if no image provided
    if image_file is None:
        return

    predict_image_response = request_prediction(image_file)

    if predict_image_response.status_code != 200:
        st.info(predict_image_response.reason)
        return

    data = data_from_json(predict_image_response.json())

    st.info(f'The bird in your image is most likely a(n) {list(data)[0]}!')
    plot_species_prob(data)

if __name__ == '__main__':
    main()

