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
# docker build -t skripsi-backend-docker .
```

After that, run image by executing:  
Without GPU:  
```bash
# docker run -p 8889:8889 skripsi-backend-docker
```
With GPU:  
```bash
# docker run -p 8889:8889 --gpu all skripsi-backend-docker
```
