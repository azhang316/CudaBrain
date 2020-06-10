#include <cuda.h>
#include "layer.h"

#define BLOCK_SIZE 128

__device__ void Dense(float *input, float *weights, float *output)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    
}

__device__ void Sigmoid(int *input, int* output)
{

}

__device__ void FeedForward(float data[][], float weights[][], float bias[][], float output[][])
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow*BLOCK_SIZE + threadIdx.y;
    int col = blockCol*BLOCK_SIZE + threadIdx.x;

    float Cvalue = bias[row][col];

    //processes each output chunk one submatrix at a time to use shared memory optimally
    for(int m = 0; m < gridDim; m++)
    {
        //Loading submatrixes into shared memory, Bs is also transposed
        As[threadRow][threadCol] = data[row][m*BLOCK_SIZE + threadCol];
        Bs[threadCol][threadRow] = weights[m*BLOCK_SIZE + threadRow][col];
        __syncthreads();

        //Increments Cvalue for all the shared memory elements
        for(int e = 0; e < BLOCK_SIZE; e++)
            Cvalue += As[threadRow][e] * Bs[threadCol][e];
        __syncthreads();
    }

    output[row][col] = Cvalue;
}

int fit(Dense model[], float *data, float *labels, 
        const int epochs, int batch_size, int val_split)
{
    //feed forward parth, matrix multiply each input by the weights matrix
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    for(int i=0; i<sizeof(model)/sizeof(model[0]); i++)
    {
        dim3 dimGrid(model[i])
        FeedForward<<<,100>>>(model[i])
    }

    //for(int i=sizeof(model)/sizeof(model[0]), );

    return 1;
}

int main()
{
    int2 size = make_int2(100,100);
    int labelsize = 2;
    float data[size.x][size.y];
    float labels[size.x][labelsize];

    float* d_data;
    float* d_labels;
    cudaMalloc(&d_data, size.x * size.y * sizeof(float));
    cudaMalloc(&d_labels, size.x * labelsize * sizeof(float));
    cudaMemcpy(d_data, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels, size.x * labelsize * sizeof(float), cudaMemcpyHostToDevice);

    Dense l1 = Dense(d_data, 100,100, 10, 1, 1);
    Dense l2 = Dense(l1.d_output, 100,10, 1, 1, 1);

    Dense model[2] = {input, l1, l2};
   
    print(sizeof("%i",model[0]));
    print(sizeof("%i",model[2]));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(size.x/dimBlock.x);

    int epochs = 10;
    int batch_size = 100;
    int validation_split = 0.2;
    fit(model, d_data, d_labels, 
        epochs, batch_size, validation_split);

}
