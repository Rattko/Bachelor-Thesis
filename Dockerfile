# syntax=docker/dockerfile:1

FROM python:3.10

WORKDIR /thesis

# Copy necessary files
COPY src src
COPY tools tools
COPY datasets datasets
COPY setup.cfg setup.cfg
COPY pyproject.toml pyproject.toml

# Update the operating system
RUN apt-get update && apt-get clean

# Install the latest version of Swig
RUN wget -q https://sourceforge.net/projects/swig/files/latest/download -O swig.zip
RUN unzip -q swig.zip
RUN cd swigwin-3.0.12 && ./configure && make && make install

# Install all of the dependencies
RUN pip3 install -r tools/requirements.txt
RUN pip3 install -e .

# Path AutoSklearn
RUN bash tools/patch_autosklearn.sh

CMD ./tools/run_docker.sh
