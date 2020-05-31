#include <cuda.h>
#include "layer.h"

__device__ void Dense(float *input, float *weights, float *output)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;

    
}

__device__ void Sigmoid(int *input, int* output)
{

}
__device__ void 
__global__ void Fit(float *data, float *labels, 
                    int epochs, int batch_size, 
                    float validation_split)
{

}

// A and B are the input matrices to be multiplied.
// out is the output matrix, AxB
// size is the # of columns in A = # rows in B
// n is the number of ops done in each thread.
__device void MatMul(float *A, float *B, float *out, 
                    const int size, unsigned int n=32)
{
    extern __shared__ T sdata[];
    

}

int main()
{
    int2 size = (100,100);
    int labelsize = 2;
    float data[size.x][size.y];
    float labels[size.x][labelsize];
    Layer l1 = Dense(2); // has input+1 * output weights to train
    Layer l2 = Activation("Sigmoid");

    // currently only supports sequential models
    model = {l1,l2};

    d_data = cudaMalloc(&data, size.x * size.y * sizeof(float));
    d_labels = cudaMalloc(&data, size.x * labelsize * sizeof(float))
    d_model = cudaMalloc(&model, sizeof(model));

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(size.x/dimBlock.x);

    int epochs = 10;
    int batch_size = 100
    int validation_split = 0.2;
    Fit<<<dimGrid, dimBlock>>>(d_model, d_data, d_labels, 
                                epochs, batch_size, validation_split);

}
