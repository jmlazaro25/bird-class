# bird-class
#### An excersise in the development/optimization and deployment of a large machine learning model.
The dataset is at [https://www.kaggle.com/datasets/gpiosenka/100-bird-species](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

#### Recorded Demo
[![Bird Class Web App Demo](https://img.youtube.com/vi/qIFdCHsppTo/0.jpg)](https://www.youtube.com/watch?v=qIFdCHsppTo)


#### Development
In this project, we use transfer learning, starting from the EfficientNetB0 architecture to classify bird images.
This could have applications in conservation efforts for endangered or threatend species.

This is also an exercise in working with limited resources. The training dataset has over 84k images.
However, hyperparamter optimization on a large neural network with a large dataset is impractical, if at all possible, on most people's personal computing device.
I found that reducing the training dataset to only include 25 images per class (bird species) allowed the trainings to complete in a reasonable time, averaging 6 hours.
This also perfectly balanced the dataset.

Once the architecture (two additional dropout-dense layer combinations before the final dense layer for classification on top of the base EfficientNetB0 architecture),
and callbacks had been determined from a few trial runs, I used Hyperopt to determine the best optimizer and dropout parameter.
See `optim.py` for details.
The training histories were stored and a short analysis of this optimization was done in `optim_analysis.ipynb`.
The best model achieved an weighted F1-score of 88.1% on the validation dataset. I am still holding out the test dataset as I may want do futher model development.
Note that the size of this model is 16.71 MB (4,381,488 parameters).

Attached to the dataset is a pretrained model (ds_model) that achieves an F1-score of 98.97% on the test dataset.
Since I trained and optimized my models with 1/8th as much training data,
I decided to retrain the layers of ds_model not found in the base EfficientNetB0 architecture with the same data to see how it would compare.
As shown in `ds_model_ltd.ipynb`, it was unable to make any meaningful predictions with the same training configuration and training dataset as the smaller model.
Note that the size of this model is 24.37 MB (6,387,293 parameters).

#### Deployment
Once I had a model I was content with, I deployed it as follows:
  1. I made an API for the model with FastAPI.
  2. I containerized the model, its API, and their minimal environment with Docker, and pushed the image to Dockerhub.
  3. I made a frontend with Streamlit, containerized it, and pushed it to Dockerhub.
  4. I ran both containers using Docker compose on an AWS EC2 instance and configured it to accept and forward HTTP requests to the Streamlit app.

[Youtube Demo](https://youtu.be/qIFdCHsppTo)
