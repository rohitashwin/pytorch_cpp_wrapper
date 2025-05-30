```
# Pytorch CPP Wrapper

1. The main file responsible for running the inference is gpt2_inference.py

2. The C++ matmul code can be found in `matmul.cpp`. Since this now enters the C++ domain, we can easily insert calls to OpenCL, CUDA, etc, for now, I've used a simple CPU based matmul. 

3. The wrapper has two parts: The C++ side and the python side, the c++ side can be found in `pytorch_wrapper.cpp`, the python side can be found in `setup.py`. Based on the device (CUDA, MPS(apple), etc.) set the device in the C++ wrapper code. 

## How to run?

`python setup.py build_ext --inplace`

`python gpt2_inference.py`
```