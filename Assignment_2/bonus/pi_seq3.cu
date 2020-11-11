#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>
#include <sys/time.h>

#define SEED     921


#define TPB 256
#define NUM_ITER 100000000
#define NUM_THREADS  10000
#define NUM_ITER_THREADS (NUM_ITER/NUM_THREADS)


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


__global__ void count_nom(int *d_res, curandState *states){
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    double x,y,z;
    const int a = 1;

    if (idx >= NUM_THREADS) return;

    const int s_idx = threadIdx.x;
    __shared__ int s_prod[TPB];

    int seed = idx; // different seed per thread
    curand_init(seed, idx, 0, &states[idx]);


    for (int iter = 0; iter < NUM_ITER_THREADS; iter++) {
        x = curand_uniform (&states[idx]);
        y = curand_uniform (&states[idx]);
        z = sqrt((x*x) + (y*y));

        if (z <= 1.0)
        {
            s_prod[s_idx]++;
        }
    }

    if (s_idx == 0) {
    int blockSum = 0;
    for (int j = 0; j < blockDim.x; ++j) {
      blockSum += s_prod[j];
    }
    printf("Block_%d, blockSum = %d\n", blockIdx.x, blockSum);
    // Try each of two versions of adding to the accumulator
    if (ATOMIC) {
      atomicAdd(d_res, blockSum);
    }
  }

}



int main(int argc, char* argv[])
{
    double pi;
    double start_time, stop_time, diference;
    int grid = (NUM_THREADS + TPB - 1)/ TPB;
    
    int *d_res;
    int *count = (int*)malloc(sizeof(int));
    cudaMalloc(&d_res, sizeof(int));

    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    curandState *dev_random;
    cudaMalloc((void**)&dev_random, grid*TPB*sizeof(curandState));

    
    // Calculate PI following a Monte Carlo method
    start_time = cpuSecond();

    count_nom<<<grid, TPB>>>(d_res, dev_random);
    
    cudaDeviceSynchronize();

    cudaMemcpy(count, d_res,sizeof(int), cudaMemcpyDeviceToHost);

    stop_time = cpuSecond();

    diference = stop_time - start_time;

    
    // Estimate Pi and display the result
    pi = ((double)count[0] / (double)(NUM_ITER_THREADS * NUM_THREADS)) * 4.0;
    
    printf("The result is %f\n", pi);
    printf("The execution time is %f\n", diference);
    
    return 0;
}
