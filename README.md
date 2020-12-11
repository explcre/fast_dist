# KITECH README

# 1. Software Environment

DOCKER

Python 3.7.5

CUDA 10.1

Tensorflow 2.3.0

horovod 0.20

## Docker pull

docker pull hama2386/acsys:kitech_dist

## Docker start

sudo nvidia-docker run -it --name {도커 이름 지정} -v {서버측 이미지넷 폴더 위치}:/image 
--gpus all -d --ip=0.0.0.0 --privileged {도커 컨테이너 e.g. hama2386/acsys:kitech_dist}

## Docker 사용

docker exec -it {도커 id} bash 
