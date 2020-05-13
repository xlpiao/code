#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdio.h>
#include <string.h>

#define BPP 1

#define BLOCKSIZE 16
texture<unsigned char, 2> tex;

__shared__ short int Gx[3][3];
__shared__ short int Gy[3][3];

__global__ void tex_filter(unsigned char* output, int width, int height) {
  Gx[0][0] = -1;
  Gx[0][1] = 0;
  Gx[0][2] = 1;
  Gx[1][0] = -2;
  Gx[1][1] = 0;
  Gx[1][2] = 2;
  Gx[2][0] = -1;
  Gx[2][1] = 0;
  Gx[2][2] = 1;
  Gy[0][0] = 1;
  Gy[0][1] = 2;
  Gy[0][2] = 1;
  Gy[1][0] = 0;
  Gy[1][1] = 0;
  Gy[1][2] = 0;
  Gy[2][0] = -1;
  Gy[2][1] = -2;
  Gy[2][2] = -1;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int pixel = 0, pixelX = 0, pixelY = 0;

  if (x - 1 >= 0 && x + 1 < height && y - 1 >= 0 && y + 1 < width) {
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        pixelX += (int)(tex2D(tex, x + i, y + j) * Gx[i + 1][j + 1]);
        pixelY += (int)(tex2D(tex, x + i, y + j) * Gy[i + 1][j + 1]);
      }
    }
  }

  pixel = abs(pixelX) + abs(pixelY);
  pixel = (pixel < 0) ? 0 : pixel;
  pixel = (pixel > 255) ? 255 : pixel;
  output[x + y * width] = pixel;

  // Read from texture and write to global memory
  // output[y*width+x]=input[y*width+x];
}

int main(int argc, char** argv) {
  unsigned char* h_data = NULL;
  cudaArray* d_input = NULL;
  unsigned char* d_output = NULL;

  unsigned int height, width;

  int OUTPUTSIZE = atoi(argv[1]);
  unsigned int newheight = OUTPUTSIZE, newwidth = OUTPUTSIZE;

  // char file_name[]="./img/lena_grey.pgm";
  char* file_name = argv[2];
  char* image_path = sdkFindFilePath(file_name, argv[0]);
  if (image_path == 0) exit(0);

  printf("Open %s\n", image_path);
  sdkLoadPGM(image_path, &h_data, &width, &height);
  int size = height * width * sizeof(unsigned char) * BPP;
  int newsize = newheight * newwidth * sizeof(unsigned char) * BPP;

  printf("Original Image Size: [%d, %d], size: %d\n", height, width, size);
  printf("Target   Image Size: [%d, %d], size: %d\n", newheight, newwidth,
         newsize);

  checkCudaErrors(cudaMalloc((void**)&d_output, newsize));

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
  checkCudaErrors(cudaMallocArray(&d_input, &channelDesc, width, height));
  checkCudaErrors(
      cudaMemcpyToArray(d_input, 0, 0, h_data, size, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaBindTextureToArray(tex, d_input, channelDesc));

  dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
  dim3 dimGrid(newwidth / dimBlock.x, newheight / dimBlock.y, 1);

  StopWatchInterface* timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  tex_filter<<<dimGrid, dimBlock>>>(d_output, newwidth, newheight);

  checkCudaErrors(cudaThreadSynchronize());
  sdkStopTimer(&timer);
  printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
  sdkDeleteTimer(&timer);

  unsigned char* h_odata;
  h_odata = (unsigned char*)malloc(newsize);
  checkCudaErrors(
      cudaMemcpy(h_odata, d_output, newsize, cudaMemcpyDeviceToHost));

  char outputpath[1024];
  strcpy(outputpath, image_path + 6);
  strcpy(outputpath + strlen(image_path + 6) - 4, "_tex_output.pgm");
  sdkSavePGM(outputpath, h_odata, newwidth, newheight);
  printf("Wrote '%s'\n\n", outputpath);

  checkCudaErrors(cudaFreeArray(d_input));
  checkCudaErrors(cudaFree(d_output));
  free(image_path);
  free(h_data);
  free(h_odata);
}
