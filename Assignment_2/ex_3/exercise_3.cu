#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define NUM_PARTICLES 10000
#define error 1e-6



__host__ __device__ void uptdateParticle(particle *particle, int iter, int id){
    //update the velocity:
    particle.velocity[0] = (3*id + iter) % NUM_PARTICLES;
    particle.velocity[1] = (4*id + iter) % NUM_PARTICLES;
    particle.velocity[2] = (5*id + iter) % NUM_PARTICLES;

    //update the position:
    particle.position[0] = particle.position[0] + particles.velocity[0]; 
    particle.position[1] = particle.position[1] + particles.velocity[1]; 
    particle.position[2] = particle.position[2] + particles.velocity[2]; 
}

__global__ void timeStep(particle *particles, int iter){
    const int id = threadIdx.x + blockIdx.x*blockDim.d_x;
    if(id < NUM_PARTICLES){
        uptdateParticle(&particles[i], iter, id);
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}



struct particle {
    float position[3];
    float velocity[3];
};


int main( int argc, char *argv[]){

    bool bien = true;
    const int iterations = argv[1];
    const int TPB = argv[2];
    const int NUM_PARTICLES = argv[3]
    const int GRID = (NUM_PARTICLES + TPB - 1)/TPB;


    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;


    struct particle *particlesCPU = new particle[NUM_PARTICLES];
    struct particle *particlesGPU= new particle[NUM_PARTICLES];
    struct particle *resGPU = new particle[NUM_PARTICLES];


    // CPU part//

    start_CPU = cpuSecond();

    for(int i = 0; i < iterations; i++){
        for(int j = 0; i < NUM_PARTICLES; j++){
            uptdateParticle(&particlesCPU[j], i, j);
        }
    }


    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;

    // Finish CPU part

    //Start GPU part

    start_GPU = cpuSecond();
    cudaMalloc(&particlesGPU, NUM_PARTICLES * sizeof(particle))

    for(int i = 0; i < iterations; i++){
        timeStep<<<GRID, TPB>>>(&particlesGPU, i);
    }

    cudaDeviceSynchtonize();
    cudaMemcpy(resGPU, particlesGPU, NUM_PARTICLES * sizeof(particle), cudaMemcpyDeviceToHost);

    stop_GPU = cpuSecond();

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
