FROM python:3.11.6-slim

WORKDIR /code

COPY ./requirments.txt /code/requirments.txt

RUN pip install --upgrade pip; \
    pip install -r requirments.txt;

COPY ./bird_class /code/bird_class
COPY ./app.py /code/app.py

CMD ["streamlit", "run", "app.py"]
