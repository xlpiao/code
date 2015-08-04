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


int main(int argc, char* argv[])
{
	N = atol(argv[1]);
	A = (float*)malloc(N*N*sizeof(float));
	B = (float*)malloc(N*N*sizeof(float));
	C = (float*)malloc(N*N*sizeof(float));
	cpuSeqOutput = (float*)malloc(N*N*sizeof(float));
	cpuOutput = (float*)malloc(N*N*sizeof(float));
	gpuOutput = (float*)malloc(N*N*sizeof(float));

	cl_platform_id* clPlatformIDs;
	cl_uint numPlatforms;
	cl_int err;

	err = clGetPlatformIDs (0, NULL, &numPlatforms);

	// if there's a platform or more, make space for ID's  
	if ((clPlatformIDs = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id))) == NULL)
	{
		printf("Failed to allocate memory for cl_platform ID's!\n\n");
		return -3000;
	}
	// get platform info for each platform and trap the NVIDIA platform if found  
	err = clGetPlatformIDs (numPlatforms, clPlatformIDs, NULL);
	printf("Number of Platforms: \t%d\n", numPlatforms);

	char stringOfPlatform[1024];
	char gpuPlatform[] = "NVIDIA CUDA";
	char cpuPlatform[] = "Intel(R) OpenCL";

	init(A, B, C);
	pthread_t thread1, thread2;


	for(int i=0; i<numPlatforms; i++){
		err = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &stringOfPlatform, NULL);
		if(strcmp(stringOfPlatform, gpuPlatform) == 0){
			printf("GPU Platfrom name: \t%s\n", stringOfPlatform);
			int ret1 = pthread_create(&thread1, NULL, runInGPU, clPlatformIDs[i]);
		}
		if(strcmp(stringOfPlatform, cpuPlatform) == 0){
			printf("CPU Platfrom name: \t%s\n", stringOfPlatform);
			int ret2 = pthread_create(&thread2, NULL, runInCPU, clPlatformIDs[i]);
		}
	}

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	// seqGemm();

	correctnessCheck(cpuOutput, gpuOutput);

	free(A);
	free(B);
	free(C);
	free(cpuSeqOutput);
	free(cpuOutput);
	free(gpuOutput);

	return 0;
}
