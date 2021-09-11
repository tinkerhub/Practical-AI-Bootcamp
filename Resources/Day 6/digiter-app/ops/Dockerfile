FROM python:3.6-slim

RUN apt-get update 

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -e .

RUN chmod +x /app/ops/run.gunicorn.sh 

CMD /app/ops/run.gunicorn.sh

