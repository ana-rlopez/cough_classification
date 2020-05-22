.PHONY: docker-build docker-run

docker-build:  ## build a docker image to run the cough_classification repository
	docker build . -t cough_classif

docker-run: #run a docker container with coughClassif_docker directory and publish port 8888 for Jupyter notebook
	docker run -v $(pwd):/coughClassif_dir -p 8888:8888 -it cough_classif
