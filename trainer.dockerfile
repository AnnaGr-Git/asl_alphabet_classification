# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# git is needed to run DVC as we use git for version control
RUN apt-get update && apt-get install -y git

RUN dvc pull

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .dvc .dvc
COPY .git .git

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc[gs]
RUN dvc pull


ENTRYPOINT ["python", "-u", "src/models/train_model.py"]