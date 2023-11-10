import streamlit as st
from pandas import DataFrame
import altair as alt

from . import call_birds_api


CLASS_STR = 'Species'
PROB_STR = 'Probability'


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
    st.altair_chart(chart, use_container_width=True)

def run():
    st.write('Upload an image below to predict a classification for your bird image.')
    image_file = st.file_uploader(
        'Upload bird image $\leq$ 1 MiB',
        type=['.png', 'jpeg'],
    )

    # Don't do anything if no image provided
    if image_file is None:
        return

    predict_image_response = call_birds_api.request_prediction(image_file)

    if predict_image_response.status_code != 200:
        st.info(predict_image_response.reason)
        return

    data = call_birds_api.data_from_json(predict_image_response.json())

    st.info(f'The bird in your image is most likely a(n) {list(data)[0]}!')
    plot_species_prob(data)

