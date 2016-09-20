#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <sstream>
#include <fstream>

#include <pthread.h>

#define KERNEL_SRC_SIZE 1000

#define LOCAL_WORK_GROUP_X 32
#define LOCAL_WORK_GROUP_Y 32

#define ALPHA 2
#define BETA 3

float* A;
float* B;
float* C;
float* cpuSeqOutput;
float* cpuOutput;
float* gpuOutput;

int N;

void init(float *A, float *B, float *C)
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			A[i*N + j] = 1;
			B[i*N + j] = 1;
			C[i*N + j] = 1;
			cpuOutput[i*N + j] = 1;
		}
	}
}

double getTime()
{
	struct timeval tv;
	gettimeofday (&tv, NULL);
	return(tv.tv_sec + tv.tv_usec*1.0E-6);
}

void correctnessCheck(float* cpuOutput, float* gpuOutput)
{
	int correct = 0;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (cpuOutput[i*N + j] == gpuOutput[i*N + j])
			{
				correct++;
			}
		}
	}
	printf("Compare result from CPU and GPU: %d  /  %d\n", correct, N*N);

}
/*
void *runInGPU(void *arg)
{	
	cl_platform_id clPlatformID = arg;
	cl_device_id clDeviceID;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_context clContext;
	cl_kernel clKernel;
	cl_command_queue clCommandQueue;
	cl_program clProgram;

	cl_int err;

	cl_mem AObj;
	cl_mem BObj;
	cl_mem CObj;

	FILE *fp;
	char *kernelSrc;
	size_t kernelSrcSize;
	double timeStart, timeEnd;

	int n = N;
	float alpha = ALPHA;
	float beta = BETA;
	size_t localWorkSize[2], globalWorkSize[2];

	fp = fopen("gemm.cl", "r");

	kernelSrc = (char*)malloc(KERNEL_SRC_SIZE);
	kernelSrcSize = fread(kernelSrc, 1, KERNEL_SRC_SIZE, fp);
	fclose(fp);

	err = clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_GPU, 1, &clDeviceID, &numDevices);
	if(err != CL_SUCCESS) printf("Error: clGetDeviceIDs\n");
	
	clContext = clCreateContext(NULL, 1, &clDeviceID, NULL, NULL, &err);
	if(err != CL_SUCCESS) printf("Error: clCreateContext\n");

	clCommandQueue = clCreateCommandQueue(clContext, clDeviceID, 0, &err);
	if(err != CL_SUCCESS) printf("Error: clCreateCommandQueue\n");

	AObj = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
	BObj = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
	CObj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * N * N, NULL, NULL);
	if(!AObj || !BObj || !CObj) printf("Error: clCreateBuffer\n");

	err = clEnqueueWriteBuffer(clCommandQueue, AObj, CL_TRUE, 0, sizeof(float) * N * N, A, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(clCommandQueue, BObj, CL_TRUE, 0, sizeof(float) * N * N, B, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(clCommandQueue, CObj, CL_TRUE, 0, sizeof(float) * N * N, C, 0, NULL, NULL);
	if(err != CL_SUCCESS)printf("Error: clEnqueueWriteBuffer\n");
	
	clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&kernelSrc, (const size_t *)&kernelSrcSize, &err);
	if(err != CL_SUCCESS) printf("Error: clCreateProgramWithSource\n");

	err = clBuildProgram(clProgram, 1, &clDeviceID, NULL, NULL, NULL);
	if(err != CL_SUCCESS) printf("Error: clBuildProgram\n");
		
	clKernel = clCreateKernel(clProgram, "gemm", &err);
	if(err != CL_SUCCESS) printf("Error: clCreateKernel\n");

	localWorkSize[0] = LOCAL_WORK_GROUP_X;
	localWorkSize[1] = LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)LOCAL_WORK_GROUP_X)) * LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)LOCAL_WORK_GROUP_Y)) * LOCAL_WORK_GROUP_Y;
	// printf("localWorkSize: %zu  %zu\n",localWorkSize[0], localWorkSize[1]);
	// printf("globalWorkSize: %zu  %zu\n",globalWorkSize[0], globalWorkSize[1]);
	err =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&AObj);
	err |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&BObj);
	err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&CObj);
	err |= clSetKernelArg(clKernel, 3, sizeof(float), (void *)&alpha);
	err |= clSetKernelArg(clKernel, 4, sizeof(float), (void *)&beta);
	err |= clSetKernelArg(clKernel, 5, sizeof(int), (void *)&n);
	
	if(err != CL_SUCCESS) printf("Error: clSetKernelArg\n");

	timeStart = getTime();

	err = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(err != CL_SUCCESS) printf("Error: clEnqueueNDRangeKernel\n");
	clFinish(clCommandQueue);

	err = clEnqueueReadBuffer(clCommandQueue, CObj, CL_TRUE, 0, N*N*sizeof(float), gpuOutput, 0, NULL, NULL);
	if(err != CL_SUCCESS) printf("Error: clEnqueueReadBuffer\n");

	timeEnd = getTime();

	printf("GPU: \t%0.2fs\n", timeEnd - timeStart);

	err = clFlush(clCommandQueue);
	err = clReleaseKernel(clKernel);
	err = clReleaseProgram(clProgram);
	err = clReleaseMemObject(AObj);
	err = clReleaseMemObject(BObj);
	err = clReleaseMemObject(CObj);
	err = clReleaseCommandQueue(clCommandQueue);
	err = clReleaseContext(clContext);
	if(err != CL_SUCCESS) printf("Error: clRelease\n");
}

void *runInCPU(void *arg)
{
	cl_platform_id clPlatformID = arg;
	cl_device_id clDeviceID;
	cl_uint numDevices;
	cl_uint numPlatforms;
	cl_context clContext;
	cl_kernel clKernel;
	cl_command_queue clCommandQueue;
	cl_program clProgram;

	cl_int err;

	cl_mem AObj;
	cl_mem BObj;
	cl_mem CObj;

	FILE *fp;
	char *kernelSrc;
	size_t kernelSrcSize;
	double timeStart, timeEnd;

	int n = N;
	float alpha = ALPHA;
	float beta = BETA;
	size_t localWorkSize[2], globalWorkSize[2];

	fp = fopen("gemm.cl", "r");

	kernelSrc = (char*)malloc(KERNEL_SRC_SIZE);
	kernelSrcSize = fread(kernelSrc, 1, KERNEL_SRC_SIZE, fp);
	fclose(fp);

	err = clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_CPU, 1, &clDeviceID, &numDevices);
	if(err != CL_SUCCESS) printf("Error: clGetDeviceIDs\n");
	
	clContext = clCreateContext(NULL, 1, &clDeviceID, NULL, NULL, &err);
	if(err != CL_SUCCESS) printf("Error: clCreateContext\n");
 
	clCommandQueue = clCreateCommandQueue(clContext, clDeviceID, 0, &err);
	if(err != CL_SUCCESS) printf("Error: clCreateCommandQueue\n");

	AObj = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
	BObj = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
	CObj = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(float) * N * N, NULL, NULL);
	if(!AObj || !BObj || !CObj) printf("Error: clCreateBuffer\n");

	err = clEnqueueWriteBuffer(clCommandQueue, AObj, CL_TRUE, 0, sizeof(float) * N * N, A, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(clCommandQueue, BObj, CL_TRUE, 0, sizeof(float) * N * N, B, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(clCommandQueue, CObj, CL_TRUE, 0, sizeof(float) * N * N, C, 0, NULL, NULL);
	if(err != CL_SUCCESS)printf("Error: clEnqueueWriteBuffer\n");
	
	clProgram = clCreateProgramWithSource(clContext, 1, (const char **)&kernelSrc, (const size_t *)&kernelSrcSize, &err);
	if(err != CL_SUCCESS) printf("Error: clCreateProgramWithSource\n");

	err = clBuildProgram(clProgram, 1, &clDeviceID, NULL, NULL, NULL);
	if(err != CL_SUCCESS) printf("Error: clBuildProgram\n");
		
	clKernel = clCreateKernel(clProgram, "gemm", &err);
	if(err != CL_SUCCESS) printf("Error: clCreateKernel\n");

	localWorkSize[0] = LOCAL_WORK_GROUP_X;
	localWorkSize[1] = LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)N) / ((float)LOCAL_WORK_GROUP_X)) * LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)N) / ((float)LOCAL_WORK_GROUP_Y)) * LOCAL_WORK_GROUP_Y;
	// printf("localWorkSize: %zu  %zu\n",localWorkSize[0], localWorkSize[1]);
	// printf("globalWorkSize: %zu  %zu\n",globalWorkSize[0], globalWorkSize[1]);
	err =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&AObj);
	err |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&BObj);
	err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&CObj);
	err |= clSetKernelArg(clKernel, 3, sizeof(float), (void *)&alpha);
	err |= clSetKernelArg(clKernel, 4, sizeof(float), (void *)&beta);
	err |= clSetKernelArg(clKernel, 5, sizeof(int), (void *)&n);
	
	if(err != CL_SUCCESS) printf("Error: clSetKernelArg\n");

	timeStart = getTime();

	err = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(err != CL_SUCCESS) printf("Error: clEnqueueNDRangeKernel\n");
	clFinish(clCommandQueue);

	err = clEnqueueReadBuffer(clCommandQueue, CObj, CL_TRUE, 0, N*N*sizeof(float), cpuOutput, 0, NULL, NULL);
	if(err != CL_SUCCESS) printf("Error: clEnqueueReadBuffer\n");

	timeEnd = getTime();

	printf("CPU: \t%0.2fs\n", timeEnd - timeStart);

	err = clFlush(clCommandQueue);
	err = clReleaseKernel(clKernel);
	err = clReleaseProgram(clProgram);
	err = clReleaseMemObject(AObj);
	err = clReleaseMemObject(BObj);
	err = clReleaseMemObject(CObj);
	err = clReleaseCommandQueue(clCommandQueue);
	err = clReleaseContext(clContext);
	if(err != CL_SUCCESS) printf("Error: clRelease\n");
}

void *seqGemm(void *arg)
{
	double timeStart, timeEnd;

	timeStart = getTime();

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cpuSeqOutput[i*N + j] *= BETA;
	
			for (int k = 0; k < N; ++k)
			{
				cpuSeqOutput[i*N + j] += ALPHA * A[i*N + k] * B[k*N + j];
			}
		}
	}

	timeEnd = getTime();

	printf("CPU (Sequential GEMM): \t%0.2fs\n", timeEnd - timeStart);
}
*/

void clPrintDevInfo(cl_device_id device) {
    char device_string[1024];

    // CL_DEVICE_NAME
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf("  CL_DEVICE_NAME: \t\t\t%s\n", device_string);

    // CL_DEVICE_VENDOR
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    printf("  CL_DEVICE_VENDOR: \t\t\t%s\n", device_string);

    // CL_DRIVER_VERSION
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
    printf("  CL_DRIVER_VERSION: \t\t\t%s\n", device_string);

    // CL_DEVICE_INFO
    cl_device_type type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if( type & CL_DEVICE_TYPE_CPU )
        printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
    if( type & CL_DEVICE_TYPE_GPU )
        printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
    if( type & CL_DEVICE_TYPE_ACCELERATOR )
        printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
    if( type & CL_DEVICE_TYPE_DEFAULT )
        printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

    // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);

    // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
    size_t workitem_dims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
    printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%lu\n", workitem_dims);

    // CL_DEVICE_MAX_WORK_ITEM_SIZES
    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%lu / %lu / %lu \n", workitem_size[0], workitem_size[1], workitem_size[2]);

    // CL_DEVICE_MAX_WORK_GROUP_SIZE
    size_t workgroup_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
    printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%lu\n", workgroup_size);

    // CL_DEVICE_MAX_CLOCK_FREQUENCY
    cl_uint clock_frequency;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", clock_frequency);

    // CL_DEVICE_ADDRESS_BITS
    cl_uint addr_bits;
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
    printf("  CL_DEVICE_ADDRESS_BITS:\t\t%u\n", addr_bits);

    // CL_DEVICE_MAX_MEM_ALLOC_SIZE
    cl_ulong max_mem_alloc_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

    // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

    // CL_DEVICE_ERROR_CORRECTION_SUPPORT
    cl_bool error_correction_support;
    clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
    printf("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", error_correction_support == CL_TRUE ? "yes" : "no");

    // CL_DEVICE_LOCAL_MEM_TYPE
    cl_device_local_mem_type local_mem_type;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
    printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

    // CL_DEVICE_LOCAL_MEM_SIZE
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

    // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));

    // CL_DEVICE_QUEUE_PROPERTIES
    cl_command_queue_properties queue_properties;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
    if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
        printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
    if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
        printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

    // CL_DEVICE_IMAGE_SUPPORT
    cl_bool image_support;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
    printf("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", image_support);

    // CL_DEVICE_MAX_READ_IMAGE_ARGS
    cl_uint max_read_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
    printf("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", max_read_image_args);

    // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
    cl_uint max_write_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
    printf("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);


    // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
    cl_uint address_bits_args;
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits_args), &address_bits_args, NULL);
    printf("  CL_DEVICE_ADRESS_BITS:\t\t%u\n", address_bits_args);

    // CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
    size_t szMaxDims[5];
    printf("  CL_DEVICE_IMAGE <dim>");
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &szMaxDims[0], NULL);
    printf("\t\t\t2D_MAX_WIDTH\t %lu\n", szMaxDims[0]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[1], NULL);
    printf("\t\t\t\t\t2D_MAX_HEIGHT\t %lu\n", szMaxDims[1]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &szMaxDims[2], NULL);
    printf("\t\t\t\t\t3D_MAX_WIDTH\t %lu\n", szMaxDims[2]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &szMaxDims[3], NULL);
    printf("\t\t\t\t\t3D_MAX_HEIGHT\t %lu\n", szMaxDims[3]);
    clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &szMaxDims[4], NULL);
    printf("\t\t\t\t\t3D_MAX_DEPTH\t %lu\n", szMaxDims[4]);

    // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
    printf("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
    cl_uint vec_width [6];
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &vec_width[0], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &vec_width[1], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &vec_width[2], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &vec_width[3], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &vec_width[4], NULL);
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &vec_width[5], NULL);
    printf("CHAR %u, SHORT %u, INT %u, FLOAT %u, DOUBLE %u\n\n",
            vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4]);

}

int main(int argc, char* argv[])
{
	// N = atol(argv[1]);
	// A = (float*)malloc(N*N*sizeof(float));
	// B = (float*)malloc(N*N*sizeof(float));
	// C = (float*)malloc(N*N*sizeof(float));
	// cpuSeqOutput = (float*)malloc(N*N*sizeof(float));
	// cpuOutput = (float*)malloc(N*N*sizeof(float));
	// gpuOutput = (float*)malloc(N*N*sizeof(float));

	cl_platform_id* clPlatformIDs;
	cl_uint numPlatforms;
	cl_device_id* clDeviceIDs;
	cl_uint numDevices;
	cl_int err;

	err = clGetPlatformIDs (0, NULL, &numPlatforms);
	printf("%d Platforms found:\n", numPlatforms);
    clPlatformIDs = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

	err = clGetPlatformIDs (numPlatforms, clPlatformIDs, NULL);
	for(int i=0; i<numPlatforms; i++){
	    char stringOfPlatform[1024];
		err = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, sizeof(stringOfPlatform), &stringOfPlatform, NULL);
		printf("\033[0;31mPlatform[%d]\033[0;37m: %s\n", i, stringOfPlatform);
        
        err = clGetDeviceIDs (clPlatformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	    printf("%d Devices found in Platfrom[%d]:\n\n", numDevices, i);
        clDeviceIDs = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

        err = clGetDeviceIDs (clPlatformIDs[i], CL_DEVICE_TYPE_ALL, numDevices, clDeviceIDs, &numDevices);        
        for(int j = 0; j < numDevices; j++ )  {
	        char stringOfDevice[1024];
            err = clGetDeviceInfo(clDeviceIDs[j], CL_DEVICE_NAME, sizeof(stringOfDevice), &stringOfDevice, NULL);
            printf("\033[0;31mDevice[%d]\033[0;37m: %s\n", j, stringOfDevice);
            // clPrintDevInfo(clDeviceIDs[j]);

            cl_device_type type;
            clGetDeviceInfo(clDeviceIDs[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            if( type & CL_DEVICE_TYPE_CPU )
                printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
            if( type & CL_DEVICE_TYPE_GPU )
                printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
            if( type & CL_DEVICE_TYPE_ACCELERATOR )
                printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
            if( type & CL_DEVICE_TYPE_DEFAULT )
                printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");
			
            // int ret2 = pthread_create(&thread2, NULL, runInCPU, clDeviceIDs[j]);
        }
        printf("\n");
		// if(strcmp(stringOfPlatform, gpuPlatform) == 0){
			// printf("GPU Platfrom name: \t%s\n", stringOfPlatform);
			// int ret1 = pthread_create(&thread1, NULL, runInGPU, clPlatformIDs[i]);
		// }
		// if(strcmp(stringOfPlatform, cpuPlatform) == 0){
			// printf("CPU Platfrom name: \t%s\n", stringOfPlatform);
			// int ret2 = pthread_create(&thread2, NULL, runInCPU, clPlatformIDs[i]);
		// }
	}

	// init(A, B, C);
	// pthread_t thread1, thread2;
	// char gpuPlatform[] = "Apple";
	// char cpuPlatform[] = "Apple";
	// char gpuPlatform[] = "NVIDIA CUDA";
	// char cpuPlatform[] = "Intel(R) OpenCL";

	// pthread_join(thread1, NULL);
	// pthread_join(thread2, NULL);

	// seqGemm();

	// correctnessCheck(cpuOutput, gpuOutput);

	// free(A);
	// free(B);
	// free(C);
	// free(cpuSeqOutput);
	// free(cpuOutput);
	// free(gpuOutput);

	return 0;
}
