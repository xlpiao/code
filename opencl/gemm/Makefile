# I tested gemm.c in Ubuntu 12.04 with OpenCL SDK and NVIDIA Driver installed
# run: ./gemm 1024

OS=$(shell uname -s)
ifeq ($(OS), Linux)
	LIBS=-lOpenCL -lpthread -lm -std=gnu99
else
	LIBS=-framework opencl -stdlib=libstdc++ -lstdc++
endif

all: gemm

gemm: gemm.c
	gcc -o gemm gemm.c $(LIBS)
clean:
	rm -f gemm
