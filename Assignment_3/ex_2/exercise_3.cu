#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define error 1e-6

#define NUM_ITERATIONS 1000
#define NUM_PARTICLES 10000
#define BLOCK_SIZE 256
#define NSTREAMS 2

struct particle {
    float position[3];
    float velocity[3];
};

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__host__ __device__ void uptdateParticle(particle *particula, int iter, int id, int num_p){
    //update the velocity:
    particula[id].velocity[0] = (3*id + iter) % num_p;
    particula[id].velocity[1] = (4*id + iter) % num_p;
    particula[id].velocity[2] = (5*id + iter) % num_p;

    //update the position:
    particula[id].position[0] = particula[id].position[0] + particula[id].velocity[0]; 
    particula[id].position[1] = particula[id].position[1] + particula[id].velocity[1]; 
    particula[id].position[2] = particula[id].position[2] + particula[id].velocity[2]; 
}

__global__ void timeStep(particle *particles, int iter, int num_p, int offset){

    const int id = offset + threadIdx.x + blockIdx.x*blockDim.x;
    if(id < num_p){
        uptdateParticle(particles, iter, id, num_p);
    }
}



int main( int argc, char *argv[]){

    
    bool bien = true;
    


    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;


    particle *particlesCPU = (particle*)malloc(NUM_PARTICLES * sizeof(particle));
    particle *particlesGPU;
    particle *resCPU;

    cudaMallocHost((void**)&resCPU, NUM_PARTICLES * sizeof(particle));
    memset(resCPU,0,NUM_PARTICLES * sizeof(particle));

    const int streamSize = NUM_PARTICLES / NSTREAMS;
    const int StreamBytes = streamSize * sizeof(particle);
    cudaStream_t stream[NSTREAMS];
    for(int i = 0; i < NSTREAMS; i++){
        cudaStreamCreate(&stream[i]);
    }
    int GRID = (streamSize + BLOCK_SIZE - 1)/BLOCK_SIZE;

    // CPU part//

    start_CPU = cpuSecond();

    for(int i = 0; i < NUM_ITERATIONS; i++){
        for(int j = 0; j < NUM_PARTICLES; j++){
            uptdateParticle(particlesCPU, i, j, NUM_PARTICLES);
        }
    };


    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;

    // Finish CPU part

    //Start GPU part

    start_GPU = cpuSecond();
    cudaMalloc((void**)&particlesGPU, NUM_PARTICLES * sizeof(particle));

    for(int s = 0; s < NSTREAMS; s++){
        
        int offset = s * streamSize;
        cudaMemcpyAsync(&particlesGPU[offset], &resCPU[offset], StreamBytes, cudaMemcpyHostToDevice, stream[s]);
    }

    for(int s = 0; s < NSTREAMS; s++){
        
        int offset = s * streamSize;
        for(int i = 0; i < NUM_ITERATIONS; i++){
            
            timeStep<<<GRID, BLOCK_SIZE, 0, stream[s]>>>(particlesGPU, i, NUM_PARTICLES, offset);
        
        }
        
    }

    for(int s = 0; s < NSTREAMS; s++){
        
        int offset = s * streamSize;
        cudaMemcpyAsync(&resCPU[offset], &particlesGPU[offset], StreamBytes, cudaMemcpyDeviceToHost, stream[s]);
    }


    
    cudaDeviceSynchronize();

    stop_GPU = cpuSecond();

    diferencia_GPU = stop_GPU - start_GPU;

    for(int i = 0; i < NUM_PARTICLES && bien; i++){
        for(int dim = 0; dim < 3; dim++){
            if(abs(particlesCPU[i].position[dim] - resCPU[i].position[dim]) > error ){
                printf("error: %d %d\n", i, dim);
                bien = false;
            }
        }
    }

    printf("NUM_ITERATIONS: %d\n", NUM_ITERATIONS);
    printf("NUM_PARTICLES: %d\n", NUM_PARTICLES);
    printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
    if(bien){
        printf("datos correctos\n");
    }else{
        printf("datos incorrectos\n");
    }
        
    cudaFree(particlesGPU);
    cudaFreeHost(resCPU);
    delete[] particlesCPU;
    
    for(int i = 0; i < NSTREAMS; i++){
        cudaStreamDestroy(stream[i]);
    }
    

    printf("Duration of the CPU: %f\n", diferencia_CPU);
    printf("Duration of the GPU: %f\n", diferencia_GPU);
    printf("--------------------------------------------\n");
    
    return 0;
}