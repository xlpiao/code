# prerequisite: OpenCL SDK and NVIDIA Driver installed
# build: make clean && make
# run:   ./gemm 512

OS=$(shell uname -s)
ifeq ($(OS), Linux)
	LIBS=-lOpenCL -lpthread -lm -std=gnu99
else
	LIBS=-framework opencl -stdlib=libc++ -lstdc++
endif

all: gemm multi_queue

gemm: gemm.cpp
	g++ -o bin.gemm gemm.cpp $(LIBS)

multi_queue: multi_queue.cpp
	g++ -o bin.multi_queue multi_queue.cpp $(LIBS)

clean:
	rm -f bin.*
