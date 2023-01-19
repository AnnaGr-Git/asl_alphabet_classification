FROM python:3.9-slim

EXPOSE $PORT

WORKDIR /src/app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn

COPY deploy_app.py deploy_app.py

CMD exec uvicorn deploy_app:app --port $PORT --host 0.0.0.0 --workers 1