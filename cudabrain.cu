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

// A and B are the input matrices to be multiplied.
// out is the output matrix, AxB
// size is the # of columns in A = # rows in B
// n is the number of ops done in each thread.
__device void MatMul(float *A, float *B, float *out, 
                    const int size, unsigned int n=32)
{
    extern __shared__ T sdata[];
    

}

int fit(Layer model[], float *data, float *labels, 
        const int epochs, int batch_size, int val_split)
{
    //feed forward parth, matrix multiply each input by the weights matrix
    for(int i=0; i<sizeof(model)/sizeof(model[0]); i++)
    {
        Matmul<<<dims here>>>(model[i].d_data, model[i].d_weights, model[i].d_output);
        BiasTerm<<<dims here>>>(model[i].offset)
    }
}

int main()
{
    int2 size = (100,100);
    int labelsize = 2;
    float data[size.x][size.y];
    float labels[size.x][labelsize];

    d_data = cudaMalloc(&data, size.x * size.y * sizeof(float));
    d_labels = cudaMalloc(&data, size.x * labelsize * sizeof(float));

    Dense input = Dense(d_data, (100, 100));
    Dense l1 = Dense(input.d_output,(100,100), 10, SIGMOID);
    Dense l2 = Dense(l1.d_output,(100,10), 1, SIGMOID);

    Layer model[3] = {input, l1, l2};
    
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(size.x/dimBlock.x);

    int epochs = 10;
    int batch_size = 100
    int validation_split = 0.2;
    fit(model, d_data, d_labels, 
        epochs, batch_size, validation_split);

}
