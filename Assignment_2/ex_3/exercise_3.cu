#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUM_PARTICLES 10000
#define error 0.05


__host__ __device__ void uptdateParticle(particle *particle, seed seeds, int iter, int id){
    //update the velocity:
    particle.velocity[0] = (seeds.x * id + iter) % NUM_PARTICLES;
    particle.velocity[1] = (seeds.y * id + iter) % NUM_PARTICLES;
    particle.velocity[2] = (seeds.z * id + iter) % NUM_PARTICLES;

    //update the position:
    particle.position[0] = particle.position[0] + particles.velocity[0]; 
    particle.position[1] = particle.position[1] + particles.velocity[1]; 
    particle.position[2] = particle.position[2] + particles.velocity[2]; 
}

__global__ void timeStep(particle *particles, seed seeds, int iter){
    const int id = threadIdx.x + blockIdx.x*blockDim.d_x;
    if(id < NUM_PARTICLES){
        uptdateParticle(&particles[i], seeds, iter, id);
    }
}


struct particle {
    float position[3];
    float velocity[3];
};

struct seed {
    int x;
    int y;
    int z;
};

int main( int argc, char *argv[]){

    bool bien = true;
    const int iterations = argv[1];
    const int TPB = argv[2];
    const int GRID = (NUM_PARTICLES + TPB - 1)/TPB;

    struct timeval time_CPU;
    struct timeval time_GPU;

    float start_GPU, stop_GPU;
    float start_CPU, stop_CPU;
    float diferencia_CPU, diferencia_GPU;


    struct seed seeds = {3,4,5};

    struct particle *particlesCPU[NUM_PARTICLES];
    struct particle *particlesGPU[NUM_PARTICLES];
    struct particle *resGPU[NUM_PARTICLES];

    // CPU part//
    gettimeofday(&timeCPU, NULL);
    start_CPU = time_CPU.tv_sec * 1000000 + time_CPU.tv_usec;

    for(int i = 0; i < iterations; i++){
        for(int j = 0; i < NUM_PARTICLES; j++){
            uptdateParticle(&particlesCPU[j], seeds, i, j);
        }
    }

    gettimeofday(&time_CPU, NULL);
    stop_CPU = time_CPU.tv_sec * 1000000 + time_CPU.tv_usec;
    diferencia_CPU = stop_CPU - start_CPU;

    // Finish CPU part

    //Start GPU part
    gettimeofday(&time_GPU, NULL);
    start_GPU = time_GPU.tv_sec * 1000000 + time_GPU.tv_usec;
    cudaMalloc(&particlesGPU, NUM_PARTICLES * sizeof(particle))

    for(int i = 0; i < iterations; i++){
        timeStep<<<GRID, TPB>>>(&particlesGPU, seeds, i);
    }

    cudaDeviceSynchtonize();
    cudaMemcpy(resGPU, particlesGPU, NUM_PARTICLES * sizeof(particle), cudaMemcpyDeviceToHost);

    gettimeofday(&time_GPU, NULL);
    stop_GPU = time_GPU.tv_sec * 1000000 + time_GPU.tv_usec;

    diferencia_GPU = stop_GPU - start_GPU;

    for(int i = 0; i < NUM_PARTICLES && bien; i++){
        for(int dim = 0; dim < 3 && bien; dim++){
            if(abs(particlesCPU[i].position[dim] - resGPU[i].position[dim]) > error ){
                bien = false;
            }
        }
    }

    if(bien){
        printf("datos correctos\n");
    }else{
        printf("datos incorrectos\n");
    }
    
    cudaFree(particlesGPU);
    delete[] resGPU;
    delete[] particlesCPU;

    printf("Duration of the CPU: %f\n", diferencia_CPU);
    printf("Duration of the GPU: %f\n", diferencia_GPU);

    return 0;


}
