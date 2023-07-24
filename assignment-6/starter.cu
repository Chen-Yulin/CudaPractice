// Histogram Equalization

#include "libwb/wb.h"

#define HISTOGRAM_LENGTH 256

// some basic transformation
__global__ void float_2_uchar(float *input, unsigned char *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        output[id] = (unsigned char) (255*input[id]); 
    }
}
__global__ void uchar_2_float(unsigned char *input, float *output, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        output[id] = (float) (input[id]/255.0);
    }
}
__global__ void RGB_2_Gray(unsigned char *input, unsigned char *output, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size) {
        output[x] = (unsigned char) (0.21*input[3*x] + 0.71*input[3*x+1] + 0.07*input[3*x+2]);
    }
}

// compute histogram
__global__ void histogram(unsigned char *input, unsigned int *output, int len) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int Histo_s[HISTOGRAM_LENGTH];

    // initialize series
    if (threadIdx.x < HISTOGRAM_LENGTH) Histo_s[threadIdx.x] = 0;

    __syncthreads();

    if (idx < len) {
        atomicAdd(&(Histo_s[input[idx]]), 1);
    }
    __syncthreads();

    if (threadIdx.x < HISTOGRAM_LENGTH) {
        atomicAdd(&(output[threadIdx.x]), Histo_s[threadIdx.x]);
    }
}

// scan operation
__global__ void scan(unsigned int *histogram, float *cdf, int size) {
    __shared__ float Histo_s[HISTOGRAM_LENGTH];

    int i = threadIdx.x;
    if (i < HISTOGRAM_LENGTH) Histo_s[i] = histogram[i];
    if (i + blockDim.x < HISTOGRAM_LENGTH) Histo_s[i+blockDim.x] = histogram[i+blockDim.x];

    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (i+1) * 2 * stride - 1;
        if (index < HISTOGRAM_LENGTH) {
            Histo_s[index] += Histo_s[index - stride];
        }
    }

    for (int stride = ceil(HISTOGRAM_LENGTH/4.0); stride > 0; stride /= 2) {
        __syncthreads();
        int idx = (i+1)*stride*2 - 1;
        if(idx + stride < HISTOGRAM_LENGTH) {
            Histo_s[idx + stride] += Histo_s[idx];
        }
    }
    __syncthreads();
    if (i < HISTOGRAM_LENGTH) cdf[i] = ((float) (Histo_s[i]*1.0)/size);
    if (i + blockDim.x < HISTOGRAM_LENGTH) cdf[i+blockDim.x] = ((float) (Histo_s[i+blockDim.x]*1.0)/size);
}

// histogram equalization function
__global__ void equalize(unsigned char *inout, float *cdf, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
    float equalized = 255.0*(cdf[inout[id]]-cdf[0])/(1.0-cdf[0]);
        inout[id] = (unsigned char) (min(max(equalized, 0.0), 255.0));
    }
}


int main(int argc, char **argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    float   *deviceImageFloat;
    unsigned char *deviceImageChar;
    unsigned char *deviceImageCharGrayScale;
    unsigned int  *deviceImageHistogram;
    float   *deviceImageCDF;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
    
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    // Allocating GPU memory
    cudaMalloc((void **)&deviceImageFloat, imageWidth*imageHeight*imageChannels*sizeof(float));
    cudaMalloc((void **)&deviceImageChar, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
    cudaMalloc((void **)&deviceImageCharGrayScale, imageWidth*imageHeight*sizeof(unsigned char));
    cudaMalloc((void **)&deviceImageHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
    cudaMalloc((void **)&deviceImageCDF, HISTOGRAM_LENGTH*sizeof(float));

    // Copy data to GPU
    cudaMemcpy(deviceImageFloat, hostInputImageData, 
                imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyHostToDevice);

    // Some preprosessing
    dim3 dimGrid1(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
    dim3 dimBlock1(512,1,1);
    float_2_uchar<<<dimGrid1,dimBlock1>>>(  deviceImageFloat,
                                            deviceImageChar,
                                            imageWidth*imageHeight*imageChannels);
    cudaDeviceSynchronize();
    dim3 dimGrid2(ceil(imageWidth*imageHeight/512.0), 1, 1);
    dim3 dimBlock2(512,1,1);
    RGB_2_Gray<<<dimGrid2,dimBlock2>>>( deviceImageChar, 
                                        deviceImageCharGrayScale, 
                                        imageWidth*imageHeight);
    cudaDeviceSynchronize();


    // histogram
    dim3 dimGrid3(ceil(imageWidth*imageHeight/256.0), 1, 1);
    dim3 dimBlock3(256,1,1);
    histogram<<<dimGrid3,dimBlock3>>>(  deviceImageCharGrayScale, 
                                        deviceImageHistogram, 
                                        imageWidth*imageHeight);
    cudaDeviceSynchronize();
    // scan
    dim3 dimGrid4(1, 1, 1);
    dim3 dimBloc4(128,1,1);
    scan<<<dimGrid4, dimBloc4>>>(deviceImageHistogram, deviceImageCDF, imageWidth*imageHeight);
    cudaDeviceSynchronize();
    // histogram equalization 
    dim3 dimGrid5(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
    dim3 dimBlock5(512,1,1);
    equalize<<<dimGrid5,dimBlock5>>>(deviceImageChar, 
                                    deviceImageCDF, imageWidth*imageHeight*imageChannels);
    cudaDeviceSynchronize();


    // cast to float
    dim3 dimGrid6(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
    dim3 dimBlock6(512,1,1);
    uchar_2_float<<<dimGrid6,dimBlock6>>>(deviceImageChar, 
                                    deviceImageFloat, imageWidth*imageHeight*imageChannels);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOutputImageData, deviceImageFloat,
                imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

    // Check Solution 
    wbImage_setData(outputImage, hostOutputImageData);

    wbSolution(args, outputImage);

    // free GPU Memory
    cudaFree(deviceImageFloat);
    cudaFree(deviceImageChar);
    cudaFree(deviceImageCharGrayScale);
    cudaFree(deviceImageHistogram);
    cudaFree(deviceImageCDF);
    // Free CPU Memory
    free(hostInputImageData);
    free(hostOutputImageData);

    return 0;
}
