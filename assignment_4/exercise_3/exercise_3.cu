// Template file for the OpenCL Assignment 4

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <sys/time.h>


#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 1000
#define BLOCK_SIZE 64
#define error 1e-6
// This is a macro for checking the error variable:>.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr,"Error: %s\n",clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char* clGetErrorString(int);




const char *mykernel = 
"typedef struct{                                         \n"
    "float position[3];                                    \n"
    "float velocity[3];                                   \n"
"}particle;                                                       \n"
"__kernel                                                 \n" 
"void kernel_upload(                                       \n"
"   const int iter,                                          \n"
"   const int nump,                                        \n"    
"   __global particle* X)                                    \n"                                      
"{int index = get_global_id(0);                           \n"
"if(index < nump) {                                                \n"
"   X[index].velocity[0] = (3*index + iter) % nump;             \n"
"   X[index].velocity[1] = (4*index + iter) % nump;            \n"
"   X[index].velocity[2] = (5*index + iter) % nump;             \n"
"   X[index].position[0] = X[index].position[0] + X[index].velocity[0];             \n"    
"   X[index].position[1] = X[index].position[1] + X[index].velocity[1];             \n"
"   X[index].position[2] = X[index].position[2] + X[index].velocity[2];             \n"
"}                                                          \n"
"}                                                        \n";

typedef struct{
    float position[3];
    float velocity[3];
}particle;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void init_Array(particle *particulas){
    for(int i = 0; i < NUM_PARTICLES; i++){
        particulas[i].position[0] = rand() % 1000;
        particulas[i].position[1] = rand() % 1000;
        particulas[i].position[2] = rand() % 1000;

        particulas[i].velocity[0] = rand() % 1000;
        particulas[i].velocity[1] = rand() % 1000;
        particulas[i].velocity[2] = rand() % 1000;
    }
}


int main(int argc, char **argv) {

    //All the variables used
    particle X[NUM_PARTICLES],  results[NUM_PARTICLES];
    float array = NUM_PARTICLES * sizeof(particle);
    int count = NUM_PARTICLES;
    bool correct = true;
    int i, j;

    cl_platform_id * platforms; cl_uint     n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
    platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

    // Find and sort devices
    cl_device_id *device_list; cl_uint n_devices;
    err = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &n_devices);CHK_ERROR(err);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*n_devices);
    err = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);CHK_ERROR(err);
    
    // Create and initialize an OpenCL context
    cl_context context = clCreateContext( NULL, n_devices, device_list, NULL, NULL, &err);CHK_ERROR(err);

    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);CHK_ERROR(err);

    if (!cmd_queue) {
        printf("Error: Failed to create a command queue!\n");
        return EXIT_FAILURE;
    }


    /* Insert your own code here */

    //program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&mykernel, NULL, &err);CHK_ERROR(err);


    //Initialize values of the arrays
 
    init_Array(X);

    //Array on GPU part
    cl_mem dx = clCreateBuffer(context, CL_MEM_READ_WRITE, array, NULL, &err);CHK_ERROR(err);


    //Copy the values of X in dx
    err = clEnqueueWriteBuffer(cmd_queue, dx, CL_TRUE, 0, array, X, 0, NULL, NULL);CHK_ERROR(err);


    //Cpu part
    double start = cpuSecond();
    for(j = 0; j < NUM_ITERATIONS; j++) {
        for(i = 0; i < NUM_PARTICLES; i++){
            X[i].velocity[0] = (3*i + j) % NUM_PARTICLES;             
            X[i].velocity[1] = (4*i + j) % NUM_PARTICLES;            
            X[i].velocity[2] = (5*i + j) % NUM_PARTICLES;             
            X[i].position[0] = X[i].position[0] + X[i].velocity[0];                 
            X[i].position[1] = X[i].position[1] + X[i].velocity[1];            
            X[i].position[2] = X[i].position[2] + X[i].velocity[2]; 
        }
    }
    double totalCpu = cpuSecond() - start;        

    printf("Computing SAXPY on the CPU... Done\n");
    
    //build the program 
    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    if (err != CL_SUCCESS){
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device_list[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        fprintf(stderr, "Build error: %s\n", buffer);
        return 0;
    }

    //Create the kernel function
    cl_kernel kernel = clCreateKernel(program, "kernel_upload", &err);CHK_ERROR(err);
    


    //More argument, number of workitems we are going to use
    unsigned int mult = BLOCK_SIZE;
    unsigned int size = BLOCK_SIZE;
    while(mult < NUM_PARTICLES){

        mult = mult * 2;
    }
    size_t n_workitem[1] = {mult};
    size_t workgroup_size[1] = {size};

    err = clSetKernelArg(kernel, 1, sizeof(int), (void*) &count);CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &dx);CHK_ERROR(err);

    
    double st = cpuSecond();
    for(i = 0; i < NUM_ITERATIONS; i++){

        //Arguments for the Kernel function
        err = clSetKernelArg(kernel, 0, sizeof(int), (void*) &i);CHK_ERROR(err);



        //Launch kernel
        err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, n_workitem, workgroup_size, 0, NULL, NULL);CHK_ERROR(err);


        //Wait for everything to finish
        err = clFlush(cmd_queue);
        err = clFinish(cmd_queue);
    }
    err = clEnqueueReadBuffer(cmd_queue, dx, CL_TRUE, 0, array, results, 0, NULL, NULL);CHK_ERROR(err);
    double timeGpu = cpuSecond() - st;
    printf("Computing SACY on the GPU... Done\n");

    for(int i = 0; i < NUM_PARTICLES; i++){
        for(int dim = 0; dim < 3; dim++){
            if(abs(results[i].position[dim] - X[i].position[dim]) > error ){
                correct = false;
                printf("Error en %d i en dim %d, resultado %f i %f\n",i,dim,results[i].position[1],X[i].position[1]);
                break;
            }
        }
        if(!correct) {break;}
    }

    if (correct) {
        printf("Comparing the output for each implementation... Correct!\n");
    }
    else {
        printf("Comparing the output for each implementation... Incorrect!\n");
    }

    printf("NUM_PARTICLES: %d\n", NUM_PARTICLES);
    printf("BLOCK_SIZE: %d\n", BLOCK_SIZE);
    printf("NUM_ITERATIONS: %d\n", NUM_ITERATIONS);
    printf("Execution time on CPU: %f\n", totalCpu);
    printf("Execution time on GPU: %f\n", timeGpu);


    // Finally, release all that we have allocated.
    clReleaseMemObject(dx);CHK_ERROR(err);
    clReleaseProgram(program);CHK_ERROR(err);
    clReleaseKernel(kernel);CHK_ERROR(err);
    err = clReleaseCommandQueue(cmd_queue);CHK_ERROR(err);
    err = clReleaseContext(context);CHK_ERROR(err);
    free(platforms);
    free(device_list);
    
    return 0;
}



// The source for this particular version is from: https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
const char* clGetErrorString(int errorCode) {
    switch (errorCode) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -71: return "CL_INVALID_SPEC_ID";
    case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
    case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
    case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
    case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
    case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
    case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
    case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
    case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
    case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
    case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
    case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
    case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
    case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
    case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
    case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
    case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
    default: return "CL_UNKNOWN_ERROR";
    }
}

