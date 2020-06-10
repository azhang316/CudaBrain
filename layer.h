#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H

#define INPUT 0 
#define DENSE 1

#define SIGMOID 0

/* This is a header file for cudabrain, since it is compiled with nvcc, it has the CUDA extensions for C++ */ 

//class Layer
//{
//public:

/* Member Data */ 
  //  int type; // Dense, etc
    //int trainable; // 0 not trainable 1 trainable
   
    //float *d_data; //input data
    //int2 data_dims; //input data dimensions
    
    //float *d_offset;
    //float *d_weights;

    //float *d_output;
    //int units; //outputs of layer
//};
/*
class Input : public Layer
{
public:
    __host__
    Input(float *data, int2 in_data_dims)
    {
        type = INPUT;
        data_dims = in_data_dims;
        units = data_dims.y;

        cudaMalloc(&d_output, sizeof(data));
        cudaMemcpy(&d_output, data, sizeof(data), cudaMemcpyHostToDevice);
    }

    void dealloc(){
        cudaFree(d_output);
    }

};*/

class Dense //: public Layer
{
public:
    int type;
    int trainable;

    float *d_data; //input data
    int data_lenx, data_leny; //input data dimensions
    
    float *d_offset;
    float *d_weights;
    int activation;

    float *d_output;
    int units; //outputs of layer

    /* Constructor: we create this class on the CPU and memory is copied to GPU for usage*/ 
    __host__
    Dense(float *d_data, int data_lenx, int data_leny, 
          int units, int activation, int trainable)
            :d_data(d_data), data_lenx(data_lenx), data_leny(data_leny),
            units(units), activation(activation), trainable(trainable)
    { 
        type = DENSE;

        //cudaMalloc(&d_offset, units*sizeof(float));
        //cudaMalloc(&d_weights, data_dims.y*units*sizeof(float));
        //cudaMalloc(&d_output, data_dims.x*units*sizeof(float)); 
    }

    void dealloc(){
        cudaFree(d_offset);
        cudaFree(d_weights);
        cudaFree(d_output);
    }
};

#endif
