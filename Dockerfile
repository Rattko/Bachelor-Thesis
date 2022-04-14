# syntax=docker/dockerfile:1

FROM python:3.10

WORKDIR /thesis

# Update the operating system
RUN apt-get update && apt-get clean

# Install the latest version of Swig
RUN wget -q https://sourceforge.net/projects/swig/files/latest/download -O swig.zip
RUN unzip -q swig.zip
RUN cd swigwin-3.0.12 && ./configure && make && make install

# Install all of the dependencies
COPY tools/requirements.txt tools/requirements.txt
RUN pip3 install -r tools/requirements.txt

# Patch AutoSklearn
COPY tools/patch_autosklearn.sh tools/patch_autosklearn.sh
COPY tools/autosklearn.patch tools/autosklearn.patch
RUN bash tools/patch_autosklearn.sh

# Copy source files and datasets
COPY src src
COPY datasets datasets
COPY setup.cfg setup.cfg
COPY pyproject.toml pyproject.toml

# Install src directory as an editable package
RUN pip3 install -e .

COPY tools/run_docker.sh tools/run_docker.sh
CMD ./tools/run_docker.sh
