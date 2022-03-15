FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	git \
	wget \
	sudo \
	ninja-build \
	build-essential \
	software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y --no-install-recommends \
	python3.8 \
	python3.8-dev \
	python3-pip \
	python3-setuptools \
	python3-distutils \
	python3-apt
RUN ln -sv /usr/bin/python3.8 /usr/bin/python
RUN python -m pip install --upgrade pip

#RUN useradd -m --no-log-init --system  --uid ${USER_ID} deeryjac -g sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER deeryjac
#WORKDIR /home/deeryjac

#ENV PATH="/home/appuser/.local/bin:${PATH}"
#RUN wget https://bootstrap.pypa.io/get-pip.py && \
#	python3 get-pip.py && \
#	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN python -m pip install tensorboard cmake   # cmake from apt-get is too old
RUN python -m pip install torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN python -m pip install cityscapesscripts

RUN python -m pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN python -m pip install -e detectron2_repo

# create a non-root user
ARG USER_ID=3114101
ARG GROUP_ID=3114101

RUN addgroup --gid $GROUP_ID appuser
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID appuser
USER appuser
WORKDIR /home/appuser

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/appuser/detectron2_repo

# run detectron2 under user "deeryjac":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
