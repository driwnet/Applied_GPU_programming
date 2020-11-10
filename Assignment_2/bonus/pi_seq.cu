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

__global__ void count(int *d_res, curandState *states, int iterations, int num_i){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    double x, y, z;
    int count;

    if (idx >= num_i) return;

    int seed = idx; // different seed per thread
    curand_init(seed, idx, 0, &states[idx])


    for (int iter = 0; iter < iterations; iter++)
    {
        x = curand_uniform (&states[id])
        y = curand_uniform (&states[id])
        z = sqrt((x*x) + (y*y));

        if (z <= 1.0)
        {
            count++;
        }
    }
    
    atomicAdd(d_res, count);
    

}



int main(int argc, char* argv[])
{
    int count;
    TPB = argv[1];
    NUM_ITER = argv[2];
    GRID = (NUM_ITER + TPB - 1) / TPB;
    NUM_ITER_CUDA = NUM_ITER / (GRID * TPB);
    double start_time, stop_time, diference;

    int *d_res = (int*)malloc(sizeof(int));
    cudaMalloc(&d_res, sideof(int));

    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, NPB*TB*sizeof(curandState));

    
    // Calculate PI following a Monte Carlo method
    start_time = cpuSecond();

    count<<<GRID, TPB>>>(d_res, dev_random, NUM_ITER_CUDA, NUM_ITER);
    
    cudaDeviceSynchtonize();

    cudaMemcpy(count, d_res, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    stop_time = cpuSecond();

    diference = stop_time - start_time;

    
    // Estimate Pi and display the result
    pi = ((double)count / (double)NUM_ITER) * 4.0;
    
    printf("The result is %f\n", pi);
    printf("The execution time is %f\n", diference);
    
    return 0;
}

