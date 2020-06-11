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

__device__ inline float sigmoid(float val)
{
    return 1.0f/ (1.0f + exp(-val));
}

// Weight initialization is done serially, as the majority of random number
// generators are sequential by nature and C random libraries cannot be used on device.
// While there are some counter based random number
// generators, the ones I found were quite difficult to understand.
// Thus I optimized this by using the Marsaglia algorithm
void initializeWeights(float *d_weights, int size)
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

__global__ void feedForward(float data[], float weights[], float bias[], float output[], dim3 size, int activation)
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

    float Cvalue = bias[col];
    
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
        output[pos] = sigmoid(Cvalue);
    else
        output[pos] = Cvalue;
}

__device__ float mse(float *d_prediction, float *d_actual)
{
    tid = blockId.x * blockDim.x + threadId.x;
    summation of differences squared ... see optimized kernel
}

__device__ float deriv_error(float d_output, float d_actual, float d_weights )
{
    de_dout = d_output - d_actual; //previous derivative error
    dout_dnet = d_output[] * (1-output[i]);
    return de_dout * dout_dnet * sum(weights);
} 

__device__ float gradient(float *d_input, float *d_output,float *deriv_error) //used for changing weight values in update
{
    float de_dout = deriv_error(); // d_output[] - d_actual;
    float dout_dnet = d_output[] * (1-d_output[]);
    float dnet_dweight = d_input[]; // can be adapted to bias term by making this 1.

    return = de_dout * dout_dnet * dnet_dweight;
}

__device__ void update()
{
    d_weight -= learning_rate * gradient()
}

__global__ void backPropagate(float *deriv_err, float * prev_deriv_err, 
                              float *wieghts, float *output)
{
    //sum weights together with gather operation
    //use map operation to multiply by d_output[]*(1-output[i])*prev_deriv_error[i]
}

int fit(Dense model[], float *data, float *labels,
        const int num_layers, const int epochs, int batch_size, int val_split)
{
    float* d_data;
    float* d_labels;
    int last = num_layers - 1;
    int data_size = model[0].size.x * model[0].size.y * sizeof(float);
    int label_size = model[last].size.x * model[last].size.z * sizeof(float);
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_labels, label_size);
    cudaMemcpy(d_labels, labels, label_size, cudaMemcpyHostToDevice);

    
    model[0].d_data = d_data;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    printf("Randomly initializing weights ...");
    for(int i=0; i<num_layers; i++)
    {
        initializeWeights(model[i].d_weights, model[i].size.y * model[i].size.z);
        initializeWeights(model[i].d_bias, model[i].size.z);
    }
    printf("Done.\n");

    //feed forward parth, matrix multiply each input by the weights matrix
    for(int i=0; i<num_layers; i++)
    {
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid((model[i].size.x - 1) / BLOCK_SIZE + 1,
                     (model[i].size.z - 1) / BLOCK_SIZE + 1);
        printf("feedforward gridsize: %d,%d\n", dimGrid.x, dimGrid.y);
        feedForward<<<dimGrid, dimBlock>>>(model[i].d_data, model[i].d_weights, 
                model[i].d_bias, model[i].d_output, 
                model[i].size, model[i].activation);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    float *prediction = (float*) malloc(label_size);
    cudaMemcpy(prediction, model[last].d_output, label_size, cudaMemcpyDeviceToHost);

    for(int i=0; i< label_size; i++)
        printf("%f \n", prediction[i]);

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
    int2 size = make_int2(256,256);
    int labelsize = 2;
    float *data = (float*)malloc(size.x * size.y * sizeof(float));
    float *labels = (float*)malloc(size.x * labelsize * sizeof(float));

    for(int i = 0; i < 256*256; i++)
        data[i] = 1;

    float* d_data;
    float* d_labels;

    printf("data size: %d ... inputted %d\n", sizeof(data), size.x * size.y * sizeof(float));

    cudaMalloc(&d_data, size.x * size.y * sizeof(float));
    cudaMemcpy(d_data, data, size.x * size.y * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_labels, size.x * labelsize * sizeof(float));
    cudaMemcpy(d_labels, labels, size.x * labelsize * sizeof(float), cudaMemcpyHostToDevice);

    printf("before creating dense layers \n");

    Dense l1 = Dense(d_data, 256, 256, 64, SIGMOID, 1);
    Dense l2 = Dense(l1.d_output, 256, 64, 32, SIGMOID, 1);
    Dense l3 = Dense(l2.d_output, 256, 32, 1, SIGMOID, 1);
    const int num_layers = 3;
    
    Dense model[num_layers] ={l1, l2, l3};

    int epochs = 10;
    int batch_size = 100;
    int validation_split = 0.2;
    printf("got to fit\n");
    fit(model, data, labels, 
        num_layers, epochs, batch_size, validation_split);
}
