# Convolution
- This directory includes convolution implementation with different languages, include python(pytorch, tensorflow, numpy), CPU sequential c/c++, and GPU CUDA version.
- Parameters stride, padding, dilation are also taken into account.
- 1D, 2D, and 3D convolution are all included.

# Docker
```
docker run --gpus all --rm -it -v $PWD:/work pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
cd /work
```

# Compile and Run
```
## install python extension
python setup.py install

## execute test.py
python test.py
```
