# set base image (host OS)
FROM python:3.8-slim-buster

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .


# install dependencies
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get -y install gcc
RUN apt-get -y install g++

RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory


# command to run on container start
