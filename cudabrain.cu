#include <cuda.h>
#include "layer.h"
#include <iostream>
#include <math.h>

#define BLOCK_SIZE 32

__device__ int Sigmoid(float val)
{
    return 1/ (1 + exp(-val));
}

__global__ void FeedForward(float data[], float weights[], float bias[], float output[], dim3 size, int activation)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;

    int threadRow = threadIdx.x;
    int threadCol = threadIdx.y;

    int row = blockRow*BLOCK_SIZE + threadRow;
    int col = blockCol*BLOCK_SIZE + threadCol;

    if(row > size.x || col > size.z)
        return;

    int pos = row * size.z + col;

    float Cvalue = bias[pos];

    int prevrows = row*size.z + threadCol;
    int prevcols = threadRow*size.z + col;

    //processes each output chunk one submatrix at a time to use shared memory optimally
    for(int m = 0; m < gridDim.y; m++)
    {
        //Loading submatrixes into shared memory, Bs is also transposed
        As[threadRow][threadCol] = data[prevrows + m*BLOCK_SIZE];
        Bs[threadRow][threadCol] = weights[prevcols + m*BLOCK_SIZE*size.x];
        __syncthreads();

        //Increments Cvalue for all the shared memory elements
    #pragma unroll
        for(int e = 0; e < BLOCK_SIZE; e++)
            Cvalue += As[threadRow][e] * Bs[e][threadCol];
        __syncthreads();
    }
    if(activation == SIGMOID)
        output[pos] = Sigmoid(Cvalue);
    else
        output[pos] = Cvalue;
}


int fit(Dense model[], float *data, float *labels,
        const int epochs, int batch_size, int val_split)
{
    float* d_data;
    float* d_labels;
    cudaMalloc(&d_data, model[0].size.x * model[0].size.y * sizeof(float));
    cudaMemcpy(d_data, data, model[0].size.x * model[0].size.y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_labels, sizeof(labels));
    cudaMemcpy(d_labels, labels, sizeof(labels), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
    for(int i=0; i<sizeof(model)/sizeof(model[0]); i++)
    {
        dim3 dimGrid(model[i].size.y / BLOCK_SIZE, model[i].size.z / BLOCK_SIZE);
        InitializeWeights<<<dimGrid, dimBlock>>>(model[i].d_weights);
    }

    //feed forward parth, matrix multiply each input by the weights matrix
    for(int i=0; i<sizeof(model)/sizeof(model[0]); i++)
    {
        dim3 dimGrid(model[i].size.x / BLOCK_SIZE, model[i].size.z / BLOCK_SIZE);
        FeedForward<<<dimGrid, dimBlock>>>(model[i].d_data, model[i].d_weights, model[i].d_bias, model[i].d_output, model[i].size, model[i].activation);
    }

    //for(int i=sizeof(model)/sizeof(model[0]), );

    return 1;
}


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
    dim3 dimGrid(size.x/BLOCK_SIZE, size.y/BLOCK_SIZE);

        //Use Cuda Events for timing
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        FeedForward<<<dimGrid, dimBlock>>>(d_data, d_weights, d_output, d_output, dim3(16384,16384,16384), SIGMOID);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::cout<< " Shared Memory Matrix Multiplication time =" << '\t'
                 << time << "ms" << std::endl;
}

int main()
{
    int2 size = make_int2(1024,1024);
    int labelsize = 2;
    float data[size.x * size.y];
    float labels[size.x * labelsize];

    for(int i = 0; i < 1024*1024; i++)
        data[i] = 1;

    float* d_data;
    float* d_labels;

    cudaMalloc(&d_data, size.x * size.y * sizeof(float));
    cudaMemcpy(d_data, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_labels, size.x * labelsize * sizeof(float));
    cudaMemcpy(d_labels, labels, size.x * labelsize * sizeof(float), cudaMemcpyHostToDevice);

    Dense l1 = Dense(d_data, 100,100, 10, 1, 1);
    Dense l2 = Dense(l1.d_output, 100,10, 1, 1, 1);

    Dense model[2] = {l1, l2};

    int epochs = 10;
    int batch_size = 100;
    int validation_split = 0.2;
    fit(model, data, labels, 
        epochs, batch_size, validation_split);
}
