FROM python:3.10-slim

EXPOSE $PORT

#RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /root

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install -e .

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .git .git
COPY entrypoint.sh entrypoint.sh

CMD exec uvicorn src.app.deploy_app:app --port $PORT --host 0.0.0.0 --workers 1
