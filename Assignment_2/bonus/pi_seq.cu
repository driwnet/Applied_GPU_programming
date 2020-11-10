#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>

#define SEED     921


int TPB;
int NUM_ITER;
int NUM_ITER_CUDA;
int GRID;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void count(int *d_res, curandState *states){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    double x, y, z;
    int count;

    if (idx >= NUM_ITER) return;

    int seed = idx; // different seed per thread
    curand_init(seed, id, 0, &states[id])



    x = curand_uniform (&states[id])
    y = curand_uniform (&states[id])
    z = sqrt((x*x) + (y*y));

    if (z <= 1.0)
    {
        count++;
    }

    
    atomicAdd(d_res, count);
    

}



int main(int argc, char* argv[])
{
    TPB = argv[1];
    NUM_ITER = argv[2];
    GRID = (NUM_ITER + TPB - 1) / TPB;
    NUM_ITER_CUDA = NUM_ITER / (GRID * TPB);

    int *d_res = (int*)malloc(sizeof(int));
    cudaMalloc(&d_res, sideof(int));

    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NPB*TB*sizeof(curandState));

    
    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < NUM_ITER_CUDA; iter++)
    {
        count<<<GRID, TPB>>>(d_res, dev_random);
        cudaDeviceSynchtonize();
    }


    
    // Estimate Pi and display the result
    pi = ((double)d_res / (double)NUM_ITER) * 4.0;
    
    printf("The result is %f\n", pi);
    
    return 0;
}

