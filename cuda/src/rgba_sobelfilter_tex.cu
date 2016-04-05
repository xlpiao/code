#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BPP 4

#define BLOCKSIZE 16
texture<uchar4, 2, cudaReadModeElementType> tex;

__shared__ short int Gx[3][3];
__shared__ short int Gy[3][3];



__device__ int sobel(unsigned char k00, unsigned char k01, unsigned char k02,
                     unsigned char k10, unsigned char k11, unsigned char k12,
                     unsigned char k20, unsigned char k21, unsigned char k22)
{
    Gx[0][0]=-1; Gx[0][1]=0; Gx[0][2]=1;
    Gx[1][0]=-2; Gx[1][1]=0; Gx[1][2]=2;
    Gx[2][0]=-1; Gx[2][1]=0; Gx[2][2]=1;
    Gy[0][0]=1; Gy[0][1]=2; Gy[0][2]=1;
    Gy[1][0]=0; Gy[1][1]=0; Gy[1][2]=0;
    Gy[2][0]=-1; Gy[2][1]=-2; Gy[2][2]=-1;
    int pixelX = k00*Gx[0][0] + k01*Gx[0][1] + k02*Gx[0][2] +
                 k10*Gx[1][0] + k11*Gx[1][1] + k12*Gx[1][2] +
                 k20*Gx[2][0] + k21*Gx[2][1] + k22*Gx[2][2];
    int pixelY = k00*Gy[0][0] + k01*Gy[0][1] + k02*Gy[0][2] +
                 k10*Gy[1][0] + k11*Gy[1][1] + k12*Gy[1][2] +
                 k20*Gy[2][0] + k21*Gy[2][1] + k22*Gy[2][2];
    int pixel = abs(pixelX)+abs(pixelY);
    pixel=(pixel<0)?0:pixel;
    pixel=(pixel>255)?255:pixel;

    return pixel;
}

__global__ void tex_filter(uchar4* output,int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    uchar4 k00,k01,k02,k10,k11,k12,k20,k21,k22;
    uchar4 pixel = {0,0,0,255};
    //pixel.x = 0, pixel.y = 0, pixel.z = 0, pixel.w = 255;

    if(x-1>=0 && x+1<height && y-1>=0 && y+1<width){
        k00 = tex2D(tex, x-1, y-1);
        k01 = tex2D(tex, x-1, y);
        k02 = tex2D(tex, x-1, y+1);

        k10 = tex2D(tex, x, y-1);
        k11 = tex2D(tex, x, y);
        k12 = tex2D(tex, x, y+1);

        k20 = tex2D(tex, x+1, y-1);
        k21 = tex2D(tex, x+1, y);
        k22 = tex2D(tex, x+1, y+1);

        int gradient_r = sobel(k00.x, k01.x, k02.x, k10.x, k11.x, k12.x, k20.x, k21.x, k22.x);
        int gradient_g = sobel(k00.y, k01.y, k02.y, k10.y, k11.y, k12.y, k20.y, k21.y, k22.y);
        int gradient_b = sobel(k00.z, k01.z, k02.z, k10.z, k11.z, k12.z, k20.z, k21.z, k22.z);

        //int gradient = (gradient_r + gradient_g + gradient_b) / 3;

        pixel.x = gradient_r;
        pixel.y = gradient_g;
        pixel.z = gradient_b;
        pixel.w = 255;
    }
    output[__mul24(y, width) + x] = pixel;
}


int main(int argc, char** argv)
{
    unsigned char* h_data=NULL;
    cudaArray* d_input=NULL;
    uchar4* d_output=NULL;

    unsigned int height,width;

    int OUTPUTSIZE = atoi(argv[1]);
    unsigned int newheight=OUTPUTSIZE,newwidth=OUTPUTSIZE;

    //char file_name[]="./img/lena_rgba.ppm";
    char *file_name = argv[2];
    char* image_path = sdkFindFilePath(file_name, argv[0]);
    if(image_path==0)
        exit(0);

    printf("Open %s\n",image_path);
    sdkLoadPPM4(image_path, &h_data, &width, &height);
    int size = height*width*sizeof(unsigned char)*BPP;
    int newsize=newheight*newwidth*sizeof(uchar4);

    printf("Original Image Size: [%d, %d], size: %d\n", height, width, size);
    printf("New      Image Size: [%d, %d], size: %d\n", newheight, newwidth, newsize);

    checkCudaErrors(cudaMalloc((void**)&d_output,newsize));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&d_input, &channelDesc, width, height));
    checkCudaErrors(cudaMemcpyToArray(d_input,0,0,h_data,size,cudaMemcpyHostToDevice));

    checkCudaErrors(cudaBindTextureToArray(tex,d_input,channelDesc));

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(newwidth / dimBlock.x, newheight / dimBlock.y, 1);

    StopWatchInterface *timer = NULL;
    sdkCreateTimer( &timer);
    sdkStartTimer( &timer);

    tex_filter<<<dimGrid,dimBlock>>>(d_output,newwidth,newheight);

    checkCudaErrors( cudaThreadSynchronize() );
    sdkStopTimer( &timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue( &timer));
    sdkDeleteTimer(&timer);

    unsigned char* h_odata;
    h_odata=(unsigned char*)malloc(newsize);
    checkCudaErrors(cudaMemcpy(h_odata,d_output,newsize,cudaMemcpyDeviceToHost));

    char outputpath[1024];
    strcpy(outputpath,image_path+6);
    strcpy(outputpath+strlen(image_path+6)-4,"_tex_output.ppm");
    sdkSavePPM4ub( outputpath, h_odata, newwidth, newheight);
    printf("Wrote '%s'\n\n", outputpath);

    checkCudaErrors(cudaFreeArray(d_input));
    checkCudaErrors(cudaFree(d_output));
    free(image_path);
    free(h_data);
    free(h_odata);
}
