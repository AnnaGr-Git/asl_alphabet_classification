# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# git is needed to run DVC as we use git for version control
RUN apt-get update && apt-get install -y git

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

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN pip install dvc[gs]
RUN pip install -e .
RUN ls

ENTRYPOINT ["sh", "entrypoint.sh"]
