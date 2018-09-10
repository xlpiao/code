OS=$(shell uname -s)
ifeq ($(OS), Linux)
	SRC=./src/
	INC=-I/usr/local/cuda/samples/common/inc/
	LIBPATH_OPENGL=-L../../common/lib/linux/x86_64 -L/usr/X11R6/lib -lGL -lGLU -lX11 -lXi -lXmu -lglut -lGLEW
	FLAGS=-DDEBUG 
	DBGFLAGS=-O0 -g -arch sm_13
endif

all: test cuda_memory_model one_dim multi_dim device_kernel crash crash_debug_0 crash_debug_1 crash_debug_2 resizepic_tex2D atomicAdd grey_sobelfilter_global grey_sobelfilter_local grey_sobelfilter_tex rgba_sobelfilter_global rgba_sobelfilter_local rgba_sobelfilter_tex sobelfilter

test: $(SRC)test.cu
	nvcc $(DBGFLAGS) -o bin.test $(SRC)test.cu
#print_in_kernel: print_in_kernel.cu
#	nvcc $(DBGFLAGS) -o bin.print_in_kernel print_in_kernel.cu $(INC)
cuda_memory_model: $(SRC)cuda_memory_model.cu  
	nvcc $(DBGFLAGS) -o bin.cuda_memory_model $(SRC)cuda_memory_model.cu
one_dim: $(SRC)one_dim.cu
	nvcc $(DBGFLAGS) -o bin.one_dim $(SRC)one_dim.cu
multi_dim: $(SRC)multi_dim.cu
	nvcc $(DBGFLAGS) -o bin.multi_dim $(SRC)multi_dim.cu
device_kernel: $(SRC)device_kernel.cu
	nvcc $(DBGFLAGS) -o bin.device_kernel $(SRC)device_kernel.cu
crash: $(SRC)crash.cu
	nvcc $(DBGFLAGS) -o bin.crash $(SRC)crash.cu
crash_debug_0: $(SRC)crash_debug_0.cu
	nvcc $(DBGFLAGS) -o bin.crash_debug_0 $(SRC)crash_debug_0.cu
crash_debug_1: $(SRC)crash_debug_1.cu
	nvcc $(DBGFLAGS) -o bin.crash_debug_1 $(SRC)crash_debug_1.cu
crash_debug_2: $(SRC)crash_debug_2.cu
	nvcc $(DBGFLAGS) $(FLAGS) -o bin.crash_debug_2 $(SRC)crash_debug_2.cu
resizepic_tex2D: $(SRC)resizepic_tex2D.cu
	nvcc $(DBGFLAGS) -o bin.resizepic_tex2D $(SRC)resizepic_tex2D.cu $(INC)
atomicAdd: $(SRC)atomicAdd.cu
	nvcc $(DBGFLAGS) -o bin.atomicAdd $(SRC)atomicAdd.cu 
grey_sobelfilter_global: $(SRC)grey_sobelfilter_global.cu
	nvcc $(DBGFLAGS) -o bin.grey_sobelfilter_global $(SRC)grey_sobelfilter_global.cu $(INC)
grey_sobelfilter_local: $(SRC)grey_sobelfilter_local.cu
	nvcc $(DBGFLAGS) -o bin.grey_sobelfilter_local $(SRC)grey_sobelfilter_local.cu $(INC)
grey_sobelfilter_tex: $(SRC)grey_sobelfilter_tex.cu
	nvcc $(DBGFLAGS) -o bin.grey_sobelfilter_tex $(SRC)grey_sobelfilter_tex.cu $(INC)
rgba_sobelfilter_global: $(SRC)rgba_sobelfilter_global.cu
	nvcc $(DBGFLAGS) -o bin.rgba_sobelfilter_global $(SRC)rgba_sobelfilter_global.cu $(INC)
rgba_sobelfilter_local: $(SRC)rgba_sobelfilter_local.cu
	nvcc $(DBGFLAGS) -o bin.rgba_sobelfilter_local $(SRC)rgba_sobelfilter_local.cu $(INC)
rgba_sobelfilter_tex: $(SRC)rgba_sobelfilter_tex.cu
	nvcc $(DBGFLAGS) -o bin.rgba_sobelfilter_tex $(SRC)rgba_sobelfilter_tex.cu $(INC)
sobelfilter: $(SRC)sobelfilter.cu
	nvcc $(DBGFLAGS) -o bin.sobelfilter $(SRC)sobelfilter.cu $(INC)

clean:
	rm -f bin.* *_output.pgm *_output.ppm
