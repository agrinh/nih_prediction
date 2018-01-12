FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER Agrin Hilmkil <agrin@peltarion.com>


USER root

RUN apt-get update
RUN apt-get install -y git python3-dev python3-pip tmux

ADD requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ENTRYPOINT "tmux"
CMD "bash"
