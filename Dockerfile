FROM python:3.11
# I don't think mujoco 2.3.7 is compatible with python 3.12
# @TODO: python:311-slim does not come with git, try using slim and then installing git separately.

WORKDIR /workspace

ENV MUJOCO_PATH=/workspace/mujoco

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace