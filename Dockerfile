FROM nvidia/cuda:11.5.1-runtime-ubuntu20.04

LABEL org.opencontainers.image.authors="Louis Raymond <louis.raymond001@binus.ac.id>, Albert Salim <albert.salim002@binus.ac.id>"

COPY . /opt/backend-skripsi
WORKDIR /opt/backend-skripsi

ENV DEBIAN_FRONTEND=noninteractive \
	PYTHON_PACKAGE=python3.10-venv \
	PYTHON=python3.10
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main\ndeb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" | tee /etc/apt/sources.list.d/deadsnakes.list && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
    ffmpeg \
	git \
	libcudnn8 \
	libsm6 \
	libxext6 \
	$PYTHON_PACKAGE
RUN ln -s /usr/bin/python3.10 /usr/bin/python3 && \
	ln -s /usr/bin/python3 /usr/bin/python
RUN $PYTHON -m ensurepip && \
	$PYTHON -m pip install --upgrade pip && \
	$PYTHON -m pip install git+https://github.com/pypa/virtualenv.git@20.8.1 && \
	$PYTHON -m virtualenv -p $PYTHON pypyenv && \
	. ./pypyenv/bin/activate
RUN $PYTHON -m pip install -r requirements.txt

EXPOSE 8889

CMD ["./main.py"]
