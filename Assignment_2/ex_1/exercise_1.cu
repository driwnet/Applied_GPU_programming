#include <stdio.h>
# define TPB 256

__global__ void hello(){
    const int Id = threadIdx.x + blockDim.x * blockIdx.x;
    printf("Hello World! My threadId is %d\n",Id);
}

int main(){
    int Block = 1;
    hello<<<Block, TPB>>>();
    cudaDeviceSynchronize();
    return 0;
}
