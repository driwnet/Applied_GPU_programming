#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define error 1e-6

int NUM_ITERATIONS;
int BLOCK_SIZE;
int NUM_PARTICLES;


__host__ __device__ void uptdateParticle(particle *particula, int iter, int id, int num_p){
    //update the velocity:
    particula.velocity[0] = (3*id + iter) % num_p;
    particula.velocity[1] = (4*id + iter) % num_p;
    particula.velocity[2] = (5*id + iter) % num_p;

    //update the position:
    particula.position[0] = particula.position[0] + particula.velocity[0]; 
    particula.position[1] = particula.position[1] + particula.velocity[1]; 
    particula.position[2] = particula.position[2] + particula.velocity[2]; 
}

__global__ void timeStep(particle *particles, int iter, int num_p){
    const int id = threadIdx.x + blockIdx.x*blockDim.d_x;
    if(id < num_p){
        uptdateParticle(particles[id], iter, id, num_p);
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
    NUM_ITERATIONS = argv[1];
    BLOCK_SIZE = argv[2];
    NUM_PARTICLES = argv[3];
    const int GRID = (NUM_PARTICLES + TPB - 1)/TPB;


    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;


    particle *particlesCPU = (particle*)malloc(NUM_PARTICLES * sizeof(particle));
    particle *particlesGPU;
    particle *resCPU = (particle*)malloc(NUM_PARTICLES * sizeof(particle));


    // CPU part//

    start_CPU = cpuSecond();

    for(int i = 0; i < NUM_ITERATIONS; i++){
        for(int j = 0; j < NUM_PARTICLES; j++){
            uptdateParticle(particlesCPU[j], i, j, NUM_PARTICLES);
        }
    }


    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;

    // Finish CPU part

    //Start GPU part

    start_GPU = cpuSecond();
    cudaMalloc(&particlesGPU, NUM_PARTICLES * sizeof(particle))

    for(int i = 0; i < NUM_ITERATIONS; i++){
        timeStep<<<GRID, BLOCK_SIZE>>>(&particlesGPU, i, NUM_PARTICLES);
    }

    cudaDeviceSynchtonize();
    cudaMemcpy(resCPU, particlesGPU, NUM_PARTICLES * sizeof(particle), cudaMemcpyDeviceToHost);

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
