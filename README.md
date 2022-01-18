# Skripsi Demo Backend

Watch for required dependencies in the requirements.txt.  
Run the server by executing `main.py` or run it with python3 on the shell  

## Docker Build  

The `Dockerfile` is configured to be used with CUDA support. If you don't have
CUDA on your computer, just continue to the next step. To ensure that CUDA can
be used install `nvidia-container-toolkit` on your host environment:  
On Arch Linux family using AUR (e.g. pikaur)
```bash
# pikaur -S nvidia-container-toolkit
```
On Ubuntu  
Please refer to [NVIDIA Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)  

To build the docker image:  
```bash
# docker build -t skripsi-docker-backend .
```

After that, run image by executing:  
Without GPU:  
```bash
# docker run -p 8889:8889 skripsi-docker-backend
```
With GPU:  
```bash
# docker run -p 8889:8889 --gpus all skripsi-docker-backend
```

To run the image and binding the logs, add `--mount src=/absolute/path/to/host/dir,target=/opt/backend-skripsi/logs,type=bind` parameter to the run command, like so:  
```bash
# docker run -p 8889:8889 --gpus all --mount src=/home/ccxex29/Programming/Python/skripsi/backend/dockerfs,target=/opt/backend-skripsi/logs,type=bind skripsi-docker-backend
```

