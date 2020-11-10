#include <stdio.h>
#include <stdlib.h>
#define TPB 256
#define N 10000
#define GRID (N + TPB - 1)/TPB
#define error 0.05


__global__ void saxpy(float *x, float *y, const float a){

    const int id = threadIdx.x + blockIdx.x*blockDim.d_x;
    if (id < N){
        y[id] = a*x[id] + y[id];
    }
}

int main(){
    float *x = (float*)malloc(N*sizeof(float));
    float *y = (float*)malloc(N*sizeof(float));
    float res = (float*)malloc(N*sizeof(float));
    const int a = 2;
    bool comp = true;

    float *d_x;
    float *d_y;
    cudaMalloc(&d_x, N*sideof(float));
    cudaMalloc(&d_y, N*sideof(float));

    for(int i = 0; i < N; i++){
        x[i] = rand() % 1000;
        y[i] = rand() % 1000;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    for(int i = 0;i<N,i++){
        res[i] = a*x[i] + y[i];
    }
    printf("Computing SAXPY on the CPU.. Done!\n");
    saxpy<<<GRID, TPB>>>(d_x, d_y, a);
    cudaDeviceSynchtonize();
    
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Computing SAXPY on the GPU.. Done!\n");

    for( int i = 0; i < N && comp, i++){
        if (res[i] < y[i] - error || res[i] > y[i] + error ){
            comp = false;
        }
    }
    if(comp){
        printf("Comparing the putput for each implementation.. Correct!");
    }else {
        printf("Comparing the putput for each implementation.. Incorrect!");
    }
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
