#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H

#define INPUT 0 
#define DENSE 1

#define SIGMOID 0

/* This is a header file for cudabrain, since it is compiled with nvcc, it has the CUDA extensions for C++ */ 

class Layer
{
public:

/* Member Data */ 
    int type; // Dense, etc
    int trainable; // 0 not trainable 1 trainable
};

class Input : public Layer
{
public:
    float *d_output;
    int units;

    __host__
    Input(float *data, int2 data_dims)
    {
        type = INPUT;
        dpoints = sizeof(data);

        cudaMalloc(&d_output, sizeof(data));
        cudaMemcpy(&d_output, data, sizeof(data), cudaMemcpyHostToDevice);
    }
}

class Dense : public Layer
{
public:

    float *d_data;
    
    int units;
    int activation;

    float *d_offsets;
    float *d_weights;
    float *d_output;
    
    /* Constructor: we create this class on the CPU and memory is copied to GPU for usage*/ 
    __host__
    Dense(float *d_data, int2 data_dims, 
          int units, int activation, int trainable=1){
        type = DENSE;

        this->d_data = d_data;
        this->units = units;
        this->activation = activation;
        this->trainable = trainable;

        cudaMalloc(&d_offsets, units*sizeof(float));
        cudaMalloc(&d_weights, data_dims.y*units*sizeof(float));
        cudaMalloc(&d_output, data_dims.x*units*sizeof(float)); 
    }

    void dealloc(){
        cudaFree(d_offsets);
        cudaFree(d_weights);
        cudaFree(d_output);
    }
};

#endif
