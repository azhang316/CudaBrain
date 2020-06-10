#include <cuda.h>
#include "layer.h"
#include <iostream>

#define BLOCK_SIZE 32

__device__ void Dense(float *input, float *weights, float *output)
{
//    int x = blockIdx.x*blockDim.x+threadIdx.x;
//    int y = blockIdx.y*blockDim.y+threadIdx.y;

    
}

__device__ void Sigmoid(int *input, int* output)
{

}

__global__ void FeedForward(float data[], float weights[], /*float bias[],*/ float output[])
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow*BLOCK_SIZE + threadIdx.y;
    int col = blockCol*BLOCK_SIZE + threadIdx.x;

    int pos = row * gridDim.y * BLOCK_SIZE + col;

    float Cvalue = 0;//bias[pos];

    int prevrows = row*gridDim.y*BLOCK_SIZE + threadCol;
    int prevcols = threadRow*gridDim.y*BLOCK_SIZE + col;

    //processes each output chunk one submatrix at a time to use shared memory optimally
    for(int m = 0; m < gridDim.y; m++)
    {
        //Loading submatrixes into shared memory, Bs is also transposed
        As[threadRow][threadCol] = data[prevrows + m*BLOCK_SIZE];
        Bs[threadRow][threadCol] = weights[prevcols + m*BLOCK_SIZE*gridDim.x*BLOCK_SIZE];
        __syncthreads();

        //Increments Cvalue for all the shared memory elements
    #pragma unroll
        for(int e = 0; e < BLOCK_SIZE; e++)
            Cvalue += As[threadRow][e] * Bs[e][threadCol];
        __syncthreads();
    }

    output[pos] = Cvalue;
}
/*
int fit(Dense model[], float *data, float *labels,
        const int epochs, int batch_size, int val_split)
{
    //feed forward parth, matrix multiply each input by the weights matrix
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    for(int i=0; i<sizeof(model)/sizeof(model[0]); i++)
    {
        dim3 dimGrid(model[i].data_len / BLOCK_SIZE)
        FeedForward<<<,100>>>(model[i])
    }

    //for(int i=sizeof(model)/sizeof(model[0]), );

    return 1;
}
*/


void testMatMul()
{
    int2 size = make_int2(16384,16384);
    float data[size.x * size.y];

    for(int i = 0; i < 16384*16384; i++)
        data[i] = 1;

    float* d_data;
    cudaMalloc(&d_data, size.x * size.y * sizeof(float));
    cudaMemcpy(d_data, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);

    float* d_weights;
    cudaMalloc(&d_weights, size.x * size.y * sizeof(float));
    cudaMemcpy(d_weights, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);
    
    float* d_output;
    cudaMalloc(&d_output, size.x * size.y * sizeof(float));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(size.x/BLOCK_SIZE, 16384/BLOCK_SIZE);

        //Use Cuda Events for timing
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        FeedForward<<<dimGrid, dimBlock>>>(d_data, d_weights, d_output);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::cout<< " Shared Memory Matrix Multiplication time =" << '\t'
                 << time << "ms" << std::endl;
}


int main()
{
    int2 size = make_int2(16384,16384);
    int labelsize = 2;
    float data[size.x * size.y];
    float labels[size.x * labelsize];

    for(int i = 0; i < 16384*16384; i++)
        data[i] = 1;

    float* d_data;
    float* d_labels;

    cudaMalloc(&d_data, size.x * size.y * sizeof(float));
    cudaMemcpy(d_data, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_labels, size.x * labelsize * sizeof(float));
    cudaMemcpy(d_labels, labels, size.x * labelsize * sizeof(float), cudaMemcpyHostToDevice);

    Dense l1 = Dense(d_data, 100,100, 10, 1, 1);
    Dense l2 = Dense(l1.d_output, 100,10, 1, 1, 1);

    Dense model[2] = {input, l1, l2};


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(size.x/BLOCK_SIZE, 16384/BLOCK_SIZE);

        //Use Cuda Events for timing
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        FeedForward<<<dimGrid, dimBlock>>>(d_data, d_weights, d_output);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::cout<< " Shared Memory Matrix Multiplication time =" << '\t'
                 << time << "ms" << std::endl;

    int epochs = 10;
    int batch_size = 100;
    int validation_split = 0.2;
    //fit(model, d_data, d_labels, 
    //    epochs, batch_size, validation_split);

}
