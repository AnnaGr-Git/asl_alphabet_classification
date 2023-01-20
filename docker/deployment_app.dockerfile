FROM python:3.10-slim

EXPOSE $PORT

ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /root

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY ./requirements.txt requirements.txt
COPY ./setup.py setup.py
COPY ./src/ src/
COPY .git .git
COPY ./models/ models/
COPY ./deploy_app.py deploy_app.py

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install python-multipart
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install -e .

#CMD exec uvicorn src/app/deploy_app:app --port $PORT --host 0.0.0.0 --workers 1
CMD exec uvicorn src.app.deploy_app:app --port $PORT --host 0.0.0.0 --workers 1

