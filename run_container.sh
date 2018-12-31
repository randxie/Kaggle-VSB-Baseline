docker run --entrypoint "/bin/bash" $(pwd):/project -p 8080:8080 -p 8888:8888 -it pytorch/pytorch
