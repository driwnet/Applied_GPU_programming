#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define error 1e-6
#define BLOCK_SIZE 32

///////////////////////////////////////// UTILITIES ////////////////////////////////////////////////////////////////////////////////////////////
/*
**********************************************************************
function name: init_Array
description: create a matrix with random values
parameters:
            &x pointer to a row x col Matrix
return: None
**********************************************************************
*/

void init_Array(float *x, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            x[i * n + j] = rand() % 1000;
        }
    } 
}

/*
**********************************************************************
function name: init_0_Array
description: create a matrix that cointains a high number of zero values, but every row have at least one non-zero values
parameters:
            &x pointer to a row x col Matrix
return: None
**********************************************************************
*/

void init_0_Array(float *x, int row, int col){
    bool ultimo =  false;

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){

            if(j ==  col - 1 && !ultimo){
                ultimo = true;
            }
            float r = (float) rand() / RAND_MAX;
            if(r <= 0.25){

                x[i*col + j] = rand() % 1000;
                ultimo = false;

            } else{

                if (!ultimo){

                    x[i*col + j] = 0;

                } else {

                    x[i*col + j] = rand() % 1000;
                    ultimo = false;

                }
            }
        }
    }

}

/*
**********************************************************************
function name: count_NUM
description: count the non-zero elements of a matrix
parameters:
            &x pointer to a row x col Matrix
return: the total number of non-zero elements
**********************************************************************
*/

int count_Num(float *x, int row, int col){

    int count = 0;

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(x[i*col + j] != 0){
                count++;
            }
        }
    }

    return count;
}

/*
**********************************************************************
function name: sparse_matrix
description: convert a matric in a sparse matrix format in CSR
parameters:
            &x pointer to a row x col matrix
            &rows pointer to a rows + 1 vector: points to the first element in each row and assigns it the value of its index in value
            &cols pointer to a size vector: the column to which each non-zero value belongs 
            &val pointer to a size vector: contains every non-zero value of the x matrix
return: None
**********************************************************************
*/

void sparse_matrix(float *x, int *rows, int *cols, float *val, int row, int col){

    int antes = -1;
    int count = 0;
    int count_row = 0;
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(x[i*col + j] != 0){

                if(i != antes){
                    rows[count_row] = count;
                    antes = i;
                    count_row++;
                }
                cols[count] = j;
                val[count] = x[i*col + j];
                count++;
            }
        }
    }
}

/*
**********************************************************************
function name: cpuSecond()
description: when it is called you obtain the exact time in this moment
return: the time in microseonds
**********************************************************************
*/

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

/*
**********************************************************************
function name: print_matrix
description: print a matrix compose by (float) elements
parameters:
            &a pointer to a row x col Matrix
return: None
**********************************************************************
*/

void print_matrix(float *a, int row, int col){

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            printf("%f ",a[i * col + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/*
**********************************************************************
function name: print_imatrix
description: print a matrix compose by (int) elements
parameters:
            &a pointer to a row x col Matrix
return: None
**********************************************************************
*/

void print_imatrix(int *a, int row, int col){

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            printf("%d ",a[i * col + j]);
        }
        printf("\n");
    }
    printf("\n");
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////









//////////////////////////////////////// GPU KERNEL METHODS //////////////////////////////////////////////////////////////////
/*
*********************************************************************
function name: mmatrix
description: dot product of two matrix (not only square) in GPU
parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE,(m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/

__global__ void mmatrix(float *a, float *b, float *c, int m, int n, int k){

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;


    if(col < k && row < m){

        for(int i = 0; i < n; i++){
            sum += a[row * n + i] * b[i * k + col];
        }

        c[row * k + col] = sum;
    }
}

/*
*********************************************************************
function name: gpuMatrixConv
description: implementation of the convultion  operation
parameters: 
            &a GPU device pointer to a row1 X col1 matrix (A)
            &b GPU device pointer to a row2 X col2 matrix (B)
            &c GPU device output purpose pointer to a row3 X col3 matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((col3 + BLOCK_SIZE - 1) / BLOCK_SIZE,(row3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/

__global__ void gpuMatrixConv(float *a, float *b, float *c, int row1, int col1, int row2, int col2, int row3, int col3)
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

	if (row < row3 && col < col3) {
		for (int i = 0; i < row2; i++) {
			for (int j = 0; j < col2; j++) {
                sum += a[(row + i) * col1 + col + j] * b[i * row2 + j];
                
            }
        }
		c[row * col3 + col] = sum;
	}
}

/*
*********************************************************************
function name: gpu_matrix_transpose
description: matrix transpose
parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((col + BLOCK_SIZE - 1) / BLOCK_SIZE, (row + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/

__global__ void gpuMatrixTranpose(float *a, float *b, int rows, int cols){

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows && col < cols){

        int pos_a = row * cols + col;
        int pos_b = col * rows + row;
        b[pos_b] = a[pos_a];
        
    }
}

/*
*********************************************************************
function name: gpuMatrixConv
description: implementation the matrix * vector in sparse format CSR
parameters: 
            &values GPU device pointer to a size vector (values): the non-zero values for the original matrix
            &vector GPU device pointer to a col vector (x): to multiply the non-zero values
            &rows GPU device pointer to a rows + 1 vector (row): points to the first element in each row and assigns it the value of its index in value
            &cols GPU device pointer to a col size vector (col): the column to which each non-zero value belongs 
            &res GPU device output purpose pointer to a row vector (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((col3 + BLOCK_SIZE - 1) / BLOCK_SIZE,(row3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpuMVSparse(float *values, float *vector, int *rows, int *cols,float *res, int row){

    unsigned int Id = threadIdx.x + blockDim.x * blockIdx.x;

    if(Id < row){
        for(int k = rows[Id]; k < rows[Id+1]; k++){
            res[Id] += values[k]*vector[cols[k]];
        }
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////










//////////////////////////////////////////////////// CPU METHODS AND GPU PREPARATION ///////////////////////////////////////////////////////////

/*
**********************************************************************
function name: sparse_preparation
description:    compare the CPU and GPU implementation of a matrix * vector multiplication in sparse format
                1º obtaint the size of the matrix 
                2º create the matrix 
                3º count the non-zero values
                4º tranfor into CSR format
                5º Do the operation
                6º compare the results
                Optional: print the results           
return: None
**********************************************************************
*/

void sparse_preparation(){

    int col, row, val;
    bool bien = true;

    printf("\n");
    printf("\n");
    SP1: printf("Introduce the rows of A:\n");
    fflush(stdout);
    val = scanf("%d", &row);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto SP1;
    }
    SP2: printf("Introduce the columns of A:\n");
    fflush(stdout);
    val = scanf("%d", &col);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto SP2;
    }


    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;

    float *A = (float *)malloc(row * col * sizeof(float));
    float *vector = (float *)malloc(col * sizeof(float));

    init_Array(vector, 1, col);
    init_0_Array(A, row, col);
    int size = count_Num(A, row, col);

    float *values = (float *)malloc(size * sizeof(float));
    int *rows = (int *)malloc((row + 1) * sizeof(int));
    int *cols = (int *)malloc(size * sizeof(int));

    float *res = (float *)malloc(row * sizeof(float));
    float *res_F = (float *)malloc(row * sizeof(float));

    sparse_matrix(A, rows, cols, values, row, col);
    rows[row] = size;

    int *rows_GPU;
    int *cols_GPU;
    float *values_GPU;
    float *res_GPU;
    float *vector_GPU;

    cudaMalloc(&rows_GPU, (row + 1) * sizeof(int));
    cudaMalloc(&cols_GPU, size * sizeof(int));
    cudaMalloc(&values_GPU, size * sizeof(float));
    cudaMalloc(&res_GPU, row * sizeof(float));
    cudaMalloc(&vector_GPU, col * sizeof(float));
    cudaMemset(res_GPU, 0, row * sizeof(float));

    cudaMemcpy(rows_GPU, rows, (row +1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cols_GPU, cols, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(values_GPU, values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vector_GPU, vector, col * sizeof(float), cudaMemcpyHostToDevice);
    //Start CPU Part//

    start_CPU = cpuSecond();


    for(int k = 0; k < row; k++){
        res[k] = 0;
    }
    for(int i = 0; i < row; i++){
        for(int k = rows[i]; k < rows[i + 1]; k++){

            res[i] += values[k]*vector[cols[k]];
        }
    }

    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;
    

    //Stop Cpu Part // 

    unsigned int GRID = col + BLOCK_SIZE - 1 / BLOCK_SIZE;

    
    //Init GPU part//

    start_GPU = cpuSecond();

    gpuMVSparse<<<GRID, BLOCK_SIZE>>>(values_GPU, vector_GPU, rows_GPU, cols_GPU, res_GPU, row);

    cudaDeviceSynchronize();
    cudaMemcpy(res_F, res_GPU, row * sizeof(float), cudaMemcpyDeviceToHost);

    stop_GPU = cpuSecond();

    diferencia_GPU = stop_GPU - start_GPU;

    //Stop GPU part//

    //Start Checking //
    
    for(int j = 0; j < row; j++){
        if(fabs(res_F[j] - res[j]) >= error ){
            bien = false;
            printf("Error en: %f %f\n", res_F[j], res[j]);
        }
    }


    if(bien){
        printf("Comparing the output for each implementation.. Correct!\n");
    }else {
        printf("Comparing the output for each implementation.. Incorrect!\n");
    }

    char d;
    printf("Do you want to print the matrix:\n");
    printf("YES: y  or NO: n\n");
    fflush(stdout);
    scanf(" %c", &d);
    if(d == 'y'){
        print_matrix(A,row,col);
        print_matrix(values, 1, size);
        print_imatrix(rows, 1, (col + 1));
        print_imatrix(cols, 1 ,size);
        print_matrix(res,row,1);
        print_matrix(res_F,row,1);
        fflush(stdout);
    }

    printf("Duration of the CPU: %f\n", diferencia_CPU);
    printf("Duration of the GPU: %f\n", diferencia_GPU);

    delete[] A;
    delete[] vector;
    delete[] cols;
    delete[] rows;
    delete[] res;
    delete[] res_F;
    delete[] values;
    cudaFree(values_GPU);
    cudaFree(cols_GPU);
    cudaFree(rows_GPU);
    cudaFree(res_GPU);
    cudaFree(vector_GPU);
    
}

/*
**********************************************************************
function name: transpose_preparation
description:    compare the CPU and GPU implementation of the transpose operation
                1º obtaint the size of the matrix 
                2º create the matrix 
                3º Do the operation
                4º compare the results
                Optional: print the results           
return: None
**********************************************************************
*/

void tranpose_preparation(){

    int col, row, val;
    bool bien = true;

    printf("\n");
    printf("\n");
    TR1: printf("Introduce the rows of A:\n");
    fflush(stdout);
    val = scanf("%d", &row);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto TR1;
    }
    TR2: printf("Introduce the columns of A:\n");
    fflush(stdout);
    val = scanf("%d", &col);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto TR2;
    }


    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;

    float *A = (float *)malloc(row * col * sizeof(float));
    float *res = (float *)malloc(row * col * sizeof(float));
    float *res_F = (float *)malloc(row * col * sizeof(float));



    float *A_GPU;
    float *res_GPU;

    cudaMalloc(&A_GPU, row * col * sizeof(float));
    cudaMalloc(&res_GPU, row  * col * sizeof(float));

    init_Array(A, row, col);

    cudaMemcpy(A_GPU, A, row * col * sizeof(float), cudaMemcpyHostToDevice);

    //Start CPU Part//

    start_CPU = cpuSecond();

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){

            int pos_a = i * col + j;
            int pos_res = j * row + i;

            res[pos_res] = A[pos_a]; 
        }
    }


    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;
    //Stop Cpu Part // 

    unsigned int grid_rows = (row + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_colm = (col + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimGrid(grid_colm, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    
    //Init GPU part//

    start_GPU = cpuSecond();

    gpuMatrixTranpose<<<dimGrid, dimBlock>>>(A_GPU, res_GPU, row, col);

    cudaDeviceSynchronize();
    cudaMemcpy(res_F, res_GPU, row * col * sizeof(float), cudaMemcpyDeviceToHost);

    stop_GPU = cpuSecond();

    diferencia_GPU = stop_GPU - start_GPU;

    //Stop GPU part//

    //Start Checking //

    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(fabs(res_F[i * col + j] - res[i*col + j]) >= error ){
                bien = false;
                printf("Error en: %f %f\n", res_F[i * col + j], res[i * col + j]);
            }
        }
    }

    if(bien){
        printf("Comparing the output for each implementation.. Correct!\n");
    }else {
        printf("Comparing the output for each implementation.. Incorrect!\n");
    }

    char d;
    printf("Do you want to print the differents matrix:\n");
    printf("YES: y  or NO: n\n");
    fflush(stdout);
    scanf(" %c", &d);
    if(d == 'y'){
        print_matrix(A,row,col);
        print_matrix(res,col,row);
        print_matrix(res_F,col,row);
        fflush(stdout);
    }

    printf("Duration of the CPU: %f\n", diferencia_CPU);
    printf("Duration of the GPU: %f\n", diferencia_GPU);

    cudaFreeHost(A);
    cudaFreeHost(res);
    cudaFreeHost(res_F);
    cudaFree(A_GPU);
    cudaFree(res_GPU);

}

/*
**********************************************************************
function name: conv_preparation
description:    compare the CPU and GPU implementation of the convultion operation
                1º obtaint the size of the matrix A and the square matrix B
                2º create the matrix A and B with random values 
                3º Do the operation
                4º compare the results
                Optional: print the results           
return: None
**********************************************************************
*/

void conv_preparation(){

    int col1, row1, col2, row2, col3, row3, val;
    bool bien = true;

    INTRO: printf("\n");
    printf("\n");
    CO1: printf("Introduce the rows of A:\n");
    fflush(stdout);
    val = scanf("%d", &row1);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto CO1;
    }
    CO2: printf("Introduce the colums of A:\n");
    fflush(stdout);
    val = scanf("%d", &col1);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto CO2;
    }
    CO3: printf("Introduce the rows and colums of B:\n");
    fflush(stdout);
    val = scanf("%d", &col2);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto CO3;
    }
    row2 = col2;

    if(row2 >= row1 || col2 >= col1 ) {
        printf("You must introduce again the numbers, the matrix (rows and colums) A have to be higher than B\n");
        goto INTRO;
    }

    col3 = col1 - col2 + 1;
    row3 = row1 - row2 + 1;

    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;

    float *A = (float *)malloc(row1 * col1 * sizeof(float));
    float *B = (float *)malloc(row2 * col2 * sizeof(float));
    float *res = (float *)malloc(row3 * col3 * sizeof(float));
    float *res_F = (float *)malloc(row3 * col3 * sizeof(float));



    float *A_GPU;
    float *B_GPU;
    float *res_GPU;

    cudaMalloc(&A_GPU, row1 * col1 * sizeof(float));
    cudaMalloc(&B_GPU, row2 * col2 * sizeof(float));
    cudaMalloc(&res_GPU, row3  * col3 * sizeof(float));

    init_Array(A, row1, col1);
    init_Array(B, row2, col2);

    cudaMemcpy(A_GPU, A, row1 * col1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_GPU, B, row2 * col2 * sizeof(float), cudaMemcpyHostToDevice);

    //Start CPU Part//

    start_CPU = cpuSecond();

    int i, j ,k, z;
    float sum = 0.0;

    for(i = 0; i < row3; i++){
        for(z = 0; z < col3; z++){

            sum = 0.0;

            for(j = 0; j < row2; j++){
                for(k = 0; k < col2; k++){

                    sum += A[(i + j) * col1 + z + k] * B[j * row2 + k];
                
                }
            }
            res[i * col3 + z] = sum;
        }
    }


    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;
    //Stop Cpu Part // 

    unsigned int grid_rows = (row3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_colm = (col3 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimGrid(grid_colm, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    
    //Init GPU part//

    start_GPU = cpuSecond();

    gpuMatrixConv<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, res_GPU, row1, col1, row2, col2, row3, col3);

    cudaDeviceSynchronize();
    cudaMemcpy(res_F, res_GPU, row3 * col3 * sizeof(float), cudaMemcpyDeviceToHost);

    stop_GPU = cpuSecond();

    diferencia_GPU = stop_GPU - start_GPU;

    //Stop GPU part//

    //Start Checking //

    for(int i = 0; i < row3; i++){
        for(int j = 0; j < col3; j++){
            if(fabs(res_F[i * col3 + j] - res[i*col3 + j]) >= error ){
                bien = false;
                printf("Error: %f %f\n", res_F[i * col3 + j], res[i * col3 + j]);
            }
        }
    }

    if(bien){
        printf("Comparing the output for each implementation.. Correct!\n");
    }else {
        printf("Comparing the output for each implementation.. Incorrect!\n");
    }

    char d;
    printf("Do you want to print the matrix:\n");
    printf("YES: y  or NO: n\n");
    fflush(stdout);
    scanf(" %c", &d);

    if(d == 'y'){
        print_matrix(A,row1,col1);
        print_matrix(B,row2, col2);
        print_matrix(res,row3,col3);
        print_matrix(res_F,row3,col3);
        fflush(stdout);
    }

    printf("Duration of the CPU: %f\n", diferencia_CPU);
    printf("Duration of the GPU: %f\n", diferencia_GPU);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(res);
    cudaFreeHost(res_F);
    cudaFree(A_GPU);
    cudaFree(B_GPU);
    cudaFree(res_GPU);

}

/*
**********************************************************************
function name: matrix_preparation
description:    compare the CPU and GPU implementation of the matrix multiplication
                1º obtaint the size of the matrix A and the  matrix B
                2º create the matrix A and B with random values 
                3º Do the operation
                4º compare the results
                Optional: print the results           
return: None
**********************************************************************
*/

void matrix_preparation(){

    int m, n, k, val;

    printf("\n");
    printf("\n");
    MA1: printf("Introduce the rows of A:\n");
    fflush(stdout);
    val = scanf("%d", &m);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto MA1;
    }

    MA2: printf("Introduce the columns of A:\n");
    fflush(stdout);
    val = scanf("%d", &n);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto MA2;
    }
    MA3: printf("Introduce The columns of B:\n");
    fflush(stdout);
    val = scanf("%d", &k);
    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid number, please do it again.\n");
        printf("\n");
        goto MA3;
    }

    bool bien = true;


    double start_GPU, stop_GPU;
    double start_CPU, stop_CPU;
    double diferencia_CPU, diferencia_GPU;

    float *A = (float *)malloc(m * n * sizeof(float));
    float *B = (float *)malloc(n * k * sizeof(float));
    float *res = (float *)malloc(m * k * sizeof(float));
    float *res_F = (float *)malloc(m * k * sizeof(float));

    float *A_GPU;
    float *B_GPU;
    float *res_GPU;

    cudaMalloc(&A_GPU, m * n * sizeof(float));
    cudaMalloc(&B_GPU, n * k * sizeof(float));
    cudaMalloc(&res_GPU, m  * k * sizeof(float));
    init_Array(A, m, n);
    init_Array(B, n, k);
    cudaMemcpy(A_GPU, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_GPU, B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_colm = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_colm, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //init CPU part//

    start_CPU = cpuSecond();

    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            float cont = 0.0;
            for(int z = 0; z < n; z++){
                cont += A[i * n + z] * B[z * k + j];
            }
            res[i * k + j] = cont;
        }
    }

    stop_CPU = cpuSecond();
    diferencia_CPU = stop_CPU - start_CPU;

    //init GPU Part//

    start_GPU = cpuSecond();

    mmatrix<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, res_GPU, m, n, k);

    cudaDeviceSynchronize();
    cudaMemcpy(res_F, res_GPU, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    
    stop_GPU = cpuSecond();
    diferencia_GPU = stop_GPU - start_GPU;


    //check if it is correct//
    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            if(fabs(res_F[i * k + j] - res[i*k + j]) >= error ){
                bien = false;
                break;
            }
        }
        if(!bien){break;}
    }

    if(bien){
        printf("Comparing the output for each implementation.. Correct!\n");
    }else {
        printf("Comparing the output for each implementation.. Incorrect!\n");
    }


    char d;
    printf("Do you want to print the matrix:\n");
    printf("YES: y  or NO: n\n");
    fflush(stdout);
    scanf(" %c", &d);

    if(d == 'y'){
        print_matrix(A,m,n);
        print_matrix(B,n, k);
        print_matrix(res,m,k);
        print_matrix(res_F,m,k);
        fflush(stdout);
    }

    printf("Duration of the CPU: %f\n", diferencia_CPU);
    printf("Duration of the GPU: %f\n", diferencia_GPU);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(res);
    cudaFreeHost(res_F);
    cudaFree(A_GPU);
    cudaFree(B_GPU);
    cudaFree(res_GPU);

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////











///////////////////////////////////////////////////////////// MAIN //////////////////////////////////////////////////////////////////////////////
int main( int argc, char *argv[]){
    
    int op, val;
    bool salir = false;

 
    START: printf("\n");
    printf("\n");
    printf("******************************************\n");
    printf("* Select the operation you want to do:   *\n");
    printf("*                                        *\n");
    printf("* 1. Matrix Multiplication               *\n");
    printf("* 2. Matrix Convection                   *\n");
    printf("* 3. Matrix Transpose                    *\n");
    printf("* 4. Matrix-Vector Sparse                *\n");
    printf("* 5. Exit                                *\n");
    printf("*                                        *\n");
    printf("******************************************\n");
    fflush(stdout);
    printf("Introduce the number of the operation:\n");
    fflush(stdout);
    val = scanf("%d", &op);

    if(val == 0) {

        while ( (val = getchar()) != EOF && val != '\n' );
        printf("You don't introduce a valid option, please do it again.\n");
        goto START;

    }else{
        switch(op){
            case 1:

                matrix_preparation();
                fflush(stdout);
                break;

            case 2:
            
                conv_preparation();
                fflush(stdout);
                break;

            case 3:

                tranpose_preparation();
                fflush(stdout);
                break;
            
            case 4:

                sparse_preparation();
                fflush(stdout);
                break;

            case 5:

                salir = true;
                printf("Successful Exit\n");
                fflush(stdout);
                break;

            default:
                printf("You dont select any option, please do it again\n");
                fflush(stdout);
                break;
        }
        if(!salir){goto START;}
    }
    return 0;
}