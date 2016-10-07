#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <pthread.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define KERNEL_SRC_SIZE 1000
#define LOCAL_WORK_GROUP_X 16
#define LOCAL_WORK_GROUP_Y 1
#define ALPHA 1
#define BETA 1
float* A;
float* B;
float* C;
int N;
float* seqOutput;



const char* clGetErrMsg(cl_int err)
{
    switch(err)
    {
        case CL_SUCCESS: return "[DEBUG]: No Error.";
        case CL_INVALID_MEM_OBJECT: return "[DEBUG]: Invalid memory object.";
        case CL_INVALID_ARG_INDEX: return "[DEBUG]: Invalid argument index for this kernel.";
        case CL_INVALID_ARG_VALUE: return "[DEBUG]: Invalid argument value.";
        case CL_INVALID_SAMPLER: return "[DEBUG]: Invalid sampler.";
        case CL_INVALID_ARG_SIZE: return "[DEBUG]: Invalid argument size.";
        case CL_INVALID_BUFFER_SIZE: return "[DEBUG]: Invalid buffer size.";
        case CL_INVALID_HOST_PTR: return "[DEBUG]: Invalid host pointer.";
        case CL_INVALID_DEVICE: return "[DEBUG]: Invalid device.";
        case CL_INVALID_VALUE: return "[DEBUG]: Invalid value.";
        case CL_INVALID_CONTEXT: return "[DEBUG]: Invalid Context.";
        case CL_INVALID_KERNEL: return "[DEBUG]: Invalid kernel.";
        case CL_INVALID_PROGRAM: return "[DEBUG]: Invalid program object.";
        case CL_INVALID_BINARY: return "[DEBUG]: Invalid program binary.";
        case CL_INVALID_OPERATION: return "[DEBUG]: Invalid operation.";
        case CL_INVALID_BUILD_OPTIONS: return "[DEBUG]: Invalid build options.";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "[DEBUG]: Invalid executable.";
        case CL_INVALID_COMMAND_QUEUE: return "[DEBUG]: Invalid command queue.";
        case CL_INVALID_KERNEL_ARGS: return "[DEBUG]: Invalid kernel arguments.";
        case CL_INVALID_WORK_DIMENSION: return "[DEBUG]: Invalid work dimension.";
        case CL_INVALID_WORK_GROUP_SIZE: return "[DEBUG]: Invalid work group size.";
        case CL_INVALID_WORK_ITEM_SIZE: return "[DEBUG]: Invalid work item size.";
        case CL_INVALID_GLOBAL_OFFSET: return "[DEBUG]: Invalid global offset (should be NULL).";
        case CL_OUT_OF_RESOURCES: return "[DEBUG]: Insufficient resources.";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "[DEBUG]: Could not allocate mem object.";
        case CL_INVALID_EVENT_WAIT_LIST: return "[DEBUG]: Invalid event wait list.";
        case CL_OUT_OF_HOST_MEMORY: return "[DEBUG]: Out of memory on host.";
        case CL_INVALID_KERNEL_NAME: return "[DEBUG]: Invalid kernel name.";
        case CL_INVALID_KERNEL_DEFINITION: return "[DEBUG]: Invalid kernel definition.";
        case CL_BUILD_PROGRAM_FAILURE: return "[DEBUG]: Failed to build program.";
        case -1001: return "[DEBUG]: No platforms found. (Did you put ICD files in /etc/OpenCL?)";
        default: return "[DEBUG]: Unknown error.";
    }
}

void clPrintDevInfo(cl_device_id device)
{
    char device_string[1024];

    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf("  CL_DEVICE_NAME: \t\t\t%s\n", device_string);

    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(device_string), &device_string, NULL);
    printf("  CL_DEVICE_VENDOR: \t\t\t%s\n", device_string);

    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(device_string), &device_string, NULL);
    printf("  CL_DRIVER_VERSION: \t\t\t%s\n", device_string);

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

    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", compute_units);

    size_t workitem_dims;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(workitem_dims), &workitem_dims, NULL);
    printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%lu\n", workitem_dims);

    size_t workitem_size[3];
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%lu / %lu / %lu \n", workitem_size[0], workitem_size[1], workitem_size[2]);

    size_t workgroup_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
    printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%lu\n", workgroup_size);

    cl_uint clock_frequency;
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
    printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", clock_frequency);

    cl_uint addr_bits;
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
    printf("  CL_DEVICE_ADDRESS_BITS:\t\t%u\n", addr_bits);

    cl_ulong max_mem_alloc_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
    printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

    cl_ulong mem_size;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

    cl_bool error_correction_support;
    clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
    printf("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", error_correction_support == CL_TRUE ? "yes" : "no");

    cl_device_local_mem_type local_mem_type;
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
    printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", local_mem_type == 1 ? "local" : "global");

    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(mem_size / 1024));

    clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
    printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(mem_size / 1024));

    cl_command_queue_properties queue_properties;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
    if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
        printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
    if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
        printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

    cl_bool image_support;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
    printf("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", image_support);

    cl_uint max_read_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
    printf("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", max_read_image_args);

    cl_uint max_write_image_args;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
    printf("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", max_write_image_args);


    cl_uint address_bits_args;
    clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(address_bits_args), &address_bits_args, NULL);
    printf("  CL_DEVICE_ADRESS_BITS:\t\t%u\n", address_bits_args);

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

void init(float *A, float *B, float *C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N + j] = 1;
            B[i*N + j] = 1;
            // C[i*N + j] = 0;
        }
    }
}

void printArray(float* C)
{
    int correct = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%f ", C[i*N + j]);
        }
    }
    printf("\n");
}

void correctnessCheck(float* A, float* B)
{
    int correct = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (A[i*N + j] == B[i*N + j])
            {
                correct++;
            }
        }
    }
    printf("Check Correctness: %d  /  %d\n\n", correct, N*N);
}

double getTime()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return(tv.tv_sec + tv.tv_usec*1.0E-6);
}

void seqGemm(void)
{
    double timeStart, timeEnd;

    timeStart = getTime();

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            seqOutput[i*N + j] *= BETA;

            for (int k = 0; k < N; ++k)
            {
                seqOutput[i*N + j] += ALPHA * A[i*N + k] * B[k*N + j];
            }
        }
    }
    timeEnd = getTime();
    // printArray(seqOutput);
    printf("\n\033[0;31mCPU (Sequential GEMM):\033[0;37m \t%0.2fs\n", timeEnd - timeStart);
}

void *computeInDevice(void *arg)
{
    cl_device_id clDeviceID = (cl_device_id) arg;
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

    // err = clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_CPU, 1, &clDeviceID, &numDevices);
    // if(err != CL_SUCCESS) printf("Error: clGetDeviceIDs\n");

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
    printf("localWorkSize: %zu  %zu\n",localWorkSize[0], localWorkSize[1]);
    printf("globalWorkSize: %zu  %zu\n",globalWorkSize[0], globalWorkSize[1]);

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
    // printf("%s\n", clGetErrMsg(err));
    clFinish(clCommandQueue);

    float* deviceOutput;
    deviceOutput = (float*)malloc(N*N*sizeof(float));
    err = clEnqueueReadBuffer(clCommandQueue, CObj, CL_TRUE, 0, N*N*sizeof(float), deviceOutput, 0, NULL, NULL);
    if(err != CL_SUCCESS) printf("Error: clEnqueueReadBuffer\n");

    timeEnd = getTime();
    printf("Exe. Time: \t%0.2fs\n", timeEnd - timeStart);

    // printArray(deviceOutput);

    correctnessCheck(deviceOutput, seqOutput);
    free(deviceOutput);

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

int main(int argc, char* argv[])
{
    N = atol(argv[1]);
    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*N*sizeof(float));
    C = (float*)malloc(N*N*sizeof(float));
    seqOutput = (float*)malloc(N*N*sizeof(float));

    init(A, B, C);
    seqGemm();

    cl_platform_id* clPlatformIDs;
    cl_uint numPlatforms;
    cl_device_id* clDeviceIDs;
    cl_uint numDevices;
    cl_int err;

    pthread_t threads[10];
    int ret;

    err = clGetPlatformIDs (0, NULL, &numPlatforms);
    printf("\n\033[0;37m%d Platforms Found: ", numPlatforms);
    clPlatformIDs = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

    err = clGetPlatformIDs (numPlatforms, clPlatformIDs, NULL);
    for(int i=0; i<numPlatforms; i++){
        char stringOfPlatform[1024];
        err = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, sizeof(stringOfPlatform), &stringOfPlatform, NULL);
        printf("\033[0;31mPlatform[%d]\033[0;37m: %s\n", i, stringOfPlatform);
        
        err = clGetDeviceIDs (clPlatformIDs[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        printf("(%d Devices Found in Platform[%d])\n\n", numDevices, i);
        clDeviceIDs = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

        err = clGetDeviceIDs (clPlatformIDs[i], CL_DEVICE_TYPE_ALL, numDevices, clDeviceIDs, &numDevices);        
        for(int j = 0; j < numDevices; j++ ){
            char stringOfDevice[1024];
            err = clGetDeviceInfo(clDeviceIDs[j], CL_DEVICE_NAME, sizeof(stringOfDevice), &stringOfDevice, NULL);
            printf("\033[0;31m(Platform[%d], Device[%d])\033[0;37m: %s\n", i, j, stringOfDevice);
            clPrintDevInfo(clDeviceIDs[j]);

            cl_device_type type;
            clGetDeviceInfo(clDeviceIDs[j], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
            if( type & CL_DEVICE_TYPE_CPU ){
                printf("\n----- Compute In %s -----\n", "CL_DEVICE_TYPE_CPU");
                ret = pthread_create(&threads[i+j*numDevices], NULL, computeInDevice, clDeviceIDs[j]);
                pthread_join(threads[i+j*numDevices], NULL);
            }
            else if( type & CL_DEVICE_TYPE_GPU ){
                printf("\n----- Compute In %s -----\n", "CL_DEVICE_TYPE_GPU");
                ret = pthread_create(&threads[i+j*numDevices], NULL, computeInDevice, clDeviceIDs[j]);
                pthread_join(threads[i+j*numDevices], NULL);
            }
            else if( type & CL_DEVICE_TYPE_ACCELERATOR ){
                printf("\n----- Compute In %s -----\n", "CL_DEVICE_TYPE_ACCELERATOR");
            }
            else if( type & CL_DEVICE_TYPE_DEFAULT ){
                printf("\n----- Compute In %s -----\n", "CL_DEVICE_TYPE_DEFAULT");
            }
        }
        printf("\n");
    }


    free(A);
    free(B);
    free(C);
    free(seqOutput);

    return 0;
}
