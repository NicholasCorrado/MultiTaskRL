FROM python:3.11-slim
# I don't think mujoco 2.3.7 is compatible with python 3.12

# Update package lists and install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV MUJOCO_PATH=/workspace/mujoco

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace