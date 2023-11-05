import streamlit as st


def st_model():
    st.subheader('Model')
    st.markdown("""
        At the core of this project is an Tensorflow image-classification
        neural network based on the EfficientNetB0 architecture, trained on a
        subset of the
        [100-bird-species dataset](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
        (actually contains 525 species).

        Rather than scale the training dataset up to balance the dataset,
        I only took 25 images per species. This allowed me to train several
        models until they stopped learning in a few days and on a single
        personal computer, whereas a few epochs of a single model would
        take just as long using the full training dataset while not greatly
        improving performane. The models' hyperparameters were chosen with
        Bayesian optimization via Hyperopt.
    """)

def st_deployment():
    ...

def run():
    st.header('Introduction')
    st.markdown(
        """
        This project is an expercise in machine learning (ML) development
        with limited computing resources and ML operations (MLOps).
        Below are summaries of this project's components.\n
        \- Juan\n
        """
        ':e-mail: [juan.m.lazaro.ruiz@gmail.com](mailto:juan.m.lazaro.ruiz@gmail.com)'
        ' | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/j-m-lazaro)'
        ' | <a href="https://github.com/jmlazaro25"><img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" alt="text" width="30rem"></a>',
         unsafe_allow_html=True,
    )
    st.markdown(
        '#### 1. [Model](#model)\n'
        '#### 2. [Deployment ](#deployment)\n',
        unsafe_allow_html=True
    )

    st_model()
    st_deployment()

    st.markdown(
        '[:arrow_up: Go to top](#4c0e99b9)',
        unsafe_allow_html=True
    )
