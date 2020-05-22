#base image
FROM python:3.6.9-stretch

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git ffmpeg jupyter-notebook

#install dependencies
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /coughClassif_dir
ADD . /coughClassif_dir

#exposing ports
EXPOSE 8888

# Running jupyter notebook
CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0"]
