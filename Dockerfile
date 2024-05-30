FROM python:3.9.15-slim

# Set tag to cohortbuilder
LABEL tag="cohortbuilder"

LABEL description="Docker image for the CohortBuilder pipeline."
LABEL maintainer="laurent.brock@fa2.ch"

RUN mkdir -p /tmp/miniconda-pkgs /cohortbuilder

# Set the working directory
WORKDIR /cohortbuilder

# Make important directories
RUN mkdir -p /cohortbuilder/cache/{,tmp,heyex,tools} /cohortbuilder/cache/tmp/{convert,test,upload} /cohortbuilder/cohorts /cohortbuilder/configs /cohortbuilder/data/{,check,taxonomy,tests} /cohortbuilder/logs /cohortbuilder/cohortbuilder

# Change cwd
WORKDIR /cohortbuilder/cohortbuilder

# Install the dependencies (with conda)
RUN mkdir -p /opt/miniconda3
RUN apt update && apt install -y wget ffmpeg libsm6 libxext6
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh
RUN bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3
RUN rm /opt/miniconda3/miniconda.sh

# Configure conda to use the temporary directory for package cache
RUN /opt/miniconda3/bin/conda config --add pkgs_dirs /tmp/miniconda-pkgs

# Activate the environment
RUN echo "conda activate cb" >> ~/.bashrc

# Bind port 11111
EXPOSE 11111

# Copy the cwd to the working directory, except for settings.json
COPY . .
RUN rm settings.json

# Install
RUN /opt/miniconda3/bin/conda env create -f environment.yml -y -n cb

# Clean up the temporary directory after installation
RUN rm -r /tmp/miniconda-pkgs

# We require the user to mount a keys.json at /cohortbuilder and a settings.json at /cohortbuilder/cohortbuilder

# Run the app
ENTRYPOINT ["/opt/miniconda3/envs/cb/bin/python", "/cohortbuilder/cohortbuilder/run.py"]