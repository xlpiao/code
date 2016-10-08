# prerequisite: OpenCL SDK and NVIDIA Driver installed
# build: make clean && make
# run:   ./gemm 512

OS=$(shell uname -s)
ifeq ($(OS), Linux)
	LIBS=-lOpenCL -lpthread -lm -std=gnu99
else
	LIBS=-framework opencl -stdlib=libc++ -lstdc++
endif

all: gemm

gemm: gemm.cpp
	g++ -o gemm gemm.cpp $(LIBS)
clean:
	rm -f gemm
