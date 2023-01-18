FROM python:3.10-slim

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# git is needed to run DVC as we use git for version control
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip

#RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /root

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .dvc/config .dvc/config
COPY .git .git
COPY data.dvc data.dvc
COPY entrypoint.sh entrypoint.sh
COPY app.py app.py

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN pip install dvc[gs]
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install -e .

#CMD dvc pull && python -u src/data/make_dataset.py preprocess-command --num_samples=10 && exec uvicorn app:app --port $PORT --host 0.0.0.0 --workers 1
CMD dvc pull && python -u src/data/make_dataset.py preprocess-command --num_samples=10 && exec uvicorn app:app --port $PORT --host 0.0.0.0 --workers 1
#CMD exec uvicorn app:app --port $PORT --host 0.0.0.0 --workers 1
