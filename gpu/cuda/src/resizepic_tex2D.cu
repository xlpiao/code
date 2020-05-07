#include <stdio.h>
#include <string.h>
#include "helper_cuda.h"
#include "helper_functions.h"

#ifndef _RESIZEPIC_KERNEL_H_
#define _RESIZEPIC_KERNEL_H_

texture<float,2,cudaReadModeElementType> texRef;

__global__ void resizePic(float* output,int width, int height)
{
     int x= blockIdx.x * blockDim.x + threadIdx.x;
     int y= blockIdx.y * blockDim.y + threadIdx.y;

    float u = x/(float)width;
    float v = y/(float)height;

    // Read from texture and write to global memory
    output[y*width+x]=tex2D(texRef,u,v);
}

#endif


#define TARGET_SIZE 1024
char* file_name=(char*)"lena_bw.pgm";

int main(int argc, char** argv)
{
    float* h_data=NULL;
    float* d_data=NULL;

    unsigned int height,width;        //原始大小
    unsigned int newheight=TARGET_SIZE,newwidth=TARGET_SIZE;        //拉伸大小

    //开始读取图片，使用cuda的读PGM函数
    char* image_path = sdkFindFilePath(file_name, argv[0]);
    if(image_path==0)
        exit(0);

    printf("Open %s\n",image_path);
    sdkLoadPGM(image_path, &h_data, &width, &height);
    int size = height*width*sizeof(float);
    int newsize=newheight*newwidth*sizeof(float);

    printf("Original Image Size: [%-4d, %-4d], size: %d\n", height, width, size);
    printf("Target   Image Size: [%-4d, %-4d], size: %d\n", newheight, newwidth, newsize);


    checkCudaErrors(cudaMalloc((void**)&d_data,newsize));

    //为CUDA数组分配内存，并将输入图像拷贝到内存
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,0,0,h_data,size,cudaMemcpyHostToDevice));

    //设置纹理参数
    texRef.addressMode[0]=cudaAddressModeWrap;
    texRef.addressMode[1]=cudaAddressModeWrap;
    texRef.filterMode=cudaFilterModeLinear;
    texRef.normalized=true;

    //纹理和数组绑定
    checkCudaErrors(cudaBindTextureToArray(texRef,cuArray,channelDesc));

    //开始计算
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(newwidth / dimBlock.x, newheight / dimBlock.y, 1);

    StopWatchInterface *timer = NULL;
    sdkCreateTimer( &timer);
    sdkStartTimer( &timer);

    resizePic<<<dimGrid,dimBlock>>>(d_data,newwidth,newheight);

    checkCudaErrors( cudaThreadSynchronize() );
    sdkStopTimer( &timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue( &timer));
    sdkDeleteTimer(&timer);

    //拷贝结果，并存储
    float* h_odata;
    h_odata=(float*)malloc(newsize);
    checkCudaErrors(cudaMemcpy(h_odata,d_data,newsize,cudaMemcpyDeviceToHost));

    char outputpath[1024];
    strcpy(outputpath,image_path);
    strcpy(outputpath+strlen(image_path)-4,"_output.pgm");
    sdkSavePGM( outputpath, h_odata, newwidth, newheight);
    printf("Wrote '%s'\n", outputpath);

    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(image_path);
    free(h_data);
    free(h_odata);
}
