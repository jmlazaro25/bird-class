FROM python:3.11.6-slim

WORKDIR /code

COPY ./requirments_api.txt /code/requirments_api.txt

RUN pip install --upgrade pip; \
    pip install -r requirments_api.txt;


COPY ./model /code/model
COPY ./classes.json /code/classes.json
COPY ./api.py /code/api.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
