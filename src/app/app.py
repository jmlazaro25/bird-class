import streamlit as st

from bird_class import predict_tab
from bird_class import about_tab


def main():

    st.set_page_config(
        page_title='bird-class',
        page_icon=':penguin:'
    )

    st.title(':penguin: Bird Class :student:')

    predict, about = st.tabs([
        ':crystal_ball: Predict',
        ':information_source: About'
        ])

    with predict:
        predict_tab.run()

    with about:
        about_tab.run()

if __name__ == '__main__':
    main()

