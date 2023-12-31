
#include "libwb/wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    // Calculate the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Declare shared memory arrays for the sub-matrices
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    // Accumulate the result for C[row][col]
    float sum = 0;
    for (int m = 0; m < (numAColumns - 1) / TILE_WIDTH + 1; ++m) {
        // Load the matrices from global memory to shared memory
        if (row < numARows && m * TILE_WIDTH + threadIdx.x < numAColumns)
            sharedA[threadIdx.y][threadIdx.x] = A[row * numAColumns + m * TILE_WIDTH + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0;

        if (col < numBColumns && m * TILE_WIDTH + threadIdx.y < numBRows)
            sharedB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * numBColumns + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0;

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    if (row < numCRows && col < numCColumns)
        C[row * numCColumns + col] = sum;
}

int main(int argc, char **argv) {
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set
                    // this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
    hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    wbTime_stop(Generic, "Importing data and creating memory on host");
    hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    int matrixASize = numARows * numAColumns * sizeof(float);
    int matrixBSize = numBRows * numBColumns * sizeof(float);
    int matrixCSize = numCRows * numCColumns * sizeof(float);
    cudaMalloc((void **)&deviceA, matrixASize);
    cudaMalloc((void **)&deviceB, matrixBSize);
    cudaMalloc((void **)&deviceC, matrixCSize);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, matrixASize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, matrixBSize, cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 GridDim((numCColumns-1)/16+1,(numCRows-1)/16+1,1);
    dim3 BlockDim(16,16,1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<GridDim, BlockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC, matrixCSize, cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
