// C program to multiply two square matrices. 
#include <stdio.h> 
#include <math.h>
#include <time.h>
// This function multiplies mat1[][] and mat2[][], 
// and stores the result in res[][] 
void feedForward(float *mat1, float *mat2, float *res, int N) 
{     
    int i, j, k;
    float x;     
    for (i = 0; i < N; i++)     
    {         
        for (j = 0; j < N; j++)  
        {             
            res[i*N + j] = 0;        
            for (k = 0; k < N; k++)               
                res[i* N + j] += 
                x = mat1[i*N + k]*mat2[k*N +j] + mat1[i*N +k];
                res[i*N+j] = 1/(1+exp(x));
        }     
    } 
} 
double mult(int x)
{
    float *mat1 = (float*)malloc(x*x*sizeof(float));
    float *mat2 = (float*)malloc(x*x*sizeof(float));
    float *res  = (float*)malloc(x*x*sizeof(float));
    
    clock_t start, end;      
    double cpu_time_used;       
    start = clock();       /* Do the work. */      

    feedForward(mat1,mat2,res,x);

    end = clock();      
    return ((double) (end - start)) / CLOCKS_PER_SEC; 
}

int main(){
    printf("16 took %d seconds", mult(16));
    printf("64 took %d seconds", mult(64));
    printf("256 took %d seconds", mult(256));
//    printf("2048 took %d seconds", mult(2048));
//    printf("8192 took %d seconds", mult(8192));
}
