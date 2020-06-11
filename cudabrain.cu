#include <cuda.h>
#include "layer.h"
#include <iostream>
#include <math.h>
#include <random>
#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 32


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ int Sigmoid(float val)
{
    return 1/ (1 + exp(-val));
}

// Weight initialization is done serially, as the majority of random number
// generators are sequential by nature and C random libraries cannot be used on device.
// While there are some counter based random number
// generators, the ones I found were quite difficult to understand.
// Thus I optimized this by using the Marsaglia algorithm
void InitializeWeights(float *d_weights, int size)
{
    //printf("generating %d random numbers\n", size);
    //clock_t start = clock();
    
    float *rands = (float*)malloc(size*sizeof(float));

    /* no longer using c++ default rng
    std::default_random_engine generator;
    std::normal_distribution<float> distribution{0.0,0.01};
    for (int i=0; i<size; i++)
        rands[i] = distribution(generator); */

// Implementing the Marsaglia polar method of generating random numbers on
// a normal distribution. The standard deviation that is used is 0.01
    float r1 = 1, r2 = 1, S = 1, root;
    for(int i = 0; i<size-1; i+=2)
    {
        do
        {
            //too bad rand() doesnt work on gpu device ... 
            r1 = 2.0 * rand() / (double)RAND_MAX - 1;
            r2 = 2.0 * rand() / (double)RAND_MAX - 1;
            S = r1*r1 + r2*r2;
        }while(S >=1 || S==0);
        root = sqrt(-2 * log(S) / S)/100; //sqrt(-2.0 * log(S) / S) * 0.01;
        rands[i] = r1 * root;
        rands[i+1] = r2 * root;
    }
    rands[size-1] = r2 * root;

    cudaMemcpy(d_weights, rands, size*sizeof(float), cudaMemcpyHostToDevice);

    //clock_t end = clock();
    //double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    //printf("initialize weights took %f seconds \n", cpu_time_used);
    //for(int i = 0; i < size; i++)
    //    printf("%f\n",rands[i]);
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

   // if(row > size.x || col > size.z)
   //     return;

    int pos = row * size.z + col;

    float Cvalue = bias[col];
/*
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
   // if(activation == SIGMOID)
   //     output[pos] = Sigmoid(Cvalue);
   // else
  */   output[pos] = Cvalue;
}
    

int fit(Dense model[], float *data, float *labels,
        const int num_layers, const int epochs, int batch_size, int val_split)
{
    float* d_data;
    float* d_labels;
    cudaMalloc(&d_data, model[0].size.x * model[0].size.y * sizeof(float));
    cudaMemcpy(d_data, data, model[0].size.x * model[0].size.y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_labels, sizeof(labels));
    cudaMemcpy(d_labels, labels, sizeof(labels), cudaMemcpyHostToDevice);

    
    model[0].d_data = d_data;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    printf("got to initialize weights\n");
    for(int i=0; i<num_layers; i++)
    {
        InitializeWeights(model[i].d_weights, model[i].size.y * model[i].size.z);
        InitializeWeights(model[i].d_bias, model[i].size.z);
    }

    //feed forward parth, matrix multiply each input by the weights matrix
    for(int i=0; i<num_layers; i++)
    {
        printf("%d,%d\n",model[i].size.x/BLOCK_SIZE,model[i].size.z/BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((model[i].size.x - 1) / BLOCK_SIZE + 1, (model[i].size.z - 1) / BLOCK_SIZE + 1);
        printf("feedforward gridsize: %d,%d\n", dimGrid.x, dimGrid.y);
        FeedForward<<<dimGrid, dimBlock>>>(model[i].d_data, model[i].d_weights, model[i].d_bias, 
                model[i].d_output, model[i].size, model[i].activation);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    int output_size = model[0].size.x * model[0].size.y * sizeof(float);
    float *output = (float*) malloc(output_size);
    printf("output dims: %d, %d, %d\n", model[0].size.x, model[0].size.y, sizeof(output));
    cudaMemcpy(output, d_data, output_size, cudaMemcpyDeviceToHost);
    printf("output dims: %d, %d, %d\n", model[0].size.x, model[0].units, sizeof(output));

    for(int i=0; i< 1024; i++)
        printf("%f \n", data[i]);

    return 1;
}

/*
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
*/
int main()
{
    printf("start");
    int2 size = make_int2(1024,1024);
    int labelsize = 2;
    float *data = (float*)malloc(size.x * size.y * sizeof(float));
    float *labels = (float*)malloc(size.x * labelsize * sizeof(float));

    for(int i = 0; i < 1024*1024; i++)
        data[i] = 1;

    float* d_data;
    float* d_labels;

    printf("data size: %d ... inputted %d\n", sizeof(data), size.x * size.y * sizeof(float));

    cudaMalloc(&d_data, size.x * size.y * sizeof(float));
    cudaMemcpy(d_data, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_labels, size.x * labelsize * sizeof(float));
    cudaMemcpy(d_labels, labels, size.x * labelsize * sizeof(float), cudaMemcpyHostToDevice);

    printf("before creating dense layers \n");

    Dense l1 = Dense(d_data, 1024,1024, 1024, SIGMOID, 1);
    Dense l2 = Dense(l1.d_output, 1024,1024, 128, SIGMOID, 1);
    Dense l3 = Dense(l2.d_output, 1024, 128, 2, SIGMOID, 1);
    const int num_layers = 3;
    
    Dense model[num_layers] ={l1, l2, l3};

    int epochs = 10;
    int batch_size = 100;
    int validation_split = 0.2;
    printf("got to fit\n");
    fit(model, data, labels, 
        num_layers, epochs, batch_size, validation_split);
}
