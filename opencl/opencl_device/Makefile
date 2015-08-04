OS=$(shell uname -s)
ifeq ($(OS), Linux)
	LIBS=-lOpenCL
else
	LIBS=-framework opencl -stdlib=libstdc++ -lstdc++
endif

all: clDeviceQuery

clDeviceQuery: clDeviceQuery.cpp
	g++ -o clDeviceQuery clDeviceQuery.cpp $(LIBS)

clean:
	rm -f clDeviceQuery
