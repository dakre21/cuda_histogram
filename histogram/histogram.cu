/*
* Author      : David Akre
* Date        : 10/15/17
* Description : This program will take in a file which will have a
4096x4096 matrix of numbers and compute a histogram of the number
of occurances numbers 0-9 occur in that matrix and output it to 
a file given by the second argument.
* 
* Usage       : Run './histogram <input_file.txt> <output_file.txt>
*/

#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <cuda.h>

using namespace std;

#define BLOCK_SIZE   512
#define GRID_SIZE    1
#define NUM_ELEMENTS 10

// Global declaration for FILEs
FILE* file_out;
FILE* file_in;

__global__ void calc_histogram(char* dbuff, unsigned int* dcount, unsigned size) {

    unsigned int index = threadIdx.x;
    unsigned int stride = blockDim.x;

    if (size < index) {
        return;
    }

    for (unsigned int i = index; i < size; i+=stride) {
        // Increment counter per occurances
        if (dbuff[i] == '0') {
            dcount[0] += 1;
        } else if (dbuff[i] == '1') {
            dcount[1] += 1;
        } else if (dbuff[i] == '2') {
            dcount[2] += 1;
        } else if (dbuff[i] == '3') {
            dcount[3] += 1;
        } else if (dbuff[i] == '4') {
            dcount[4] += 1;
        } else if (dbuff[i] == '5') {
            dcount[5] += 1;
        } else if (dbuff[i] == '6') {
            dcount[6] += 1;
        } else if (dbuff[i] == '7') {
            dcount[7] += 1;
        } else if (dbuff[i] == '8') {
            dcount[8] += 1;
        } else if (dbuff[i] == '9') {
            dcount[9] += 1;
        }
    }
}

void app_exit(int rc) {
    // Close files and exit
    fclose(file_in);
    fclose(file_out);
    exit(rc);
}

int verify_inputs(char* argv[]) {
    // Begin verification steps
    if (argv[1] == NULL || argv[2] == NULL) {
        fprintf(stderr, "Invalid number of inputs\n");
        return -1;
    }

    // Attempt to open the input file
    file_in = fopen(argv[1], "r");

    // Check if the file exists
    if (file_in == NULL) {
        fprintf(stderr, "Input file does not exist\n");
        return -1;
    }

    // Attempt to create the output file
    file_out = fopen(argv[2], "w+");

    // Check if the file exists
    if (file_out == NULL) {
        fprintf(stderr, "Failed to create output file\n");
        return -1;
    }

    return 0;
}

int create_histogram() {
    // Forward declarations
    int rc                    = 0;
    unsigned int size         = 0;
    char* buff;
    char* dbuff;
    unsigned int count[NUM_ELEMENTS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    unsigned int* dcount;

    char test[1] = {'a'};
    char* dtest;
    
    // Read the size of the file
    fseek(file_in, 0, SEEK_END);
    size = ftell(file_in) + 1;
    rewind(file_in);

    // Malloc space for CUDA
    cudaMalloc((void**)&dbuff, size);
    cudaMalloc((void**)&dcount, NUM_ELEMENTS);
    cudaMalloc((void**)&dtest, sizeof(char));

    // Create heap space for buffer
    buff = reinterpret_cast<char*>(malloc(size*sizeof(char)));
    memset(buff, '\0', size);

    // Read file
    fread(buff, size, sizeof(char), file_in);

    cudaMemcpy(dbuff, buff, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcount, count, NUM_ELEMENTS, cudaMemcpyHostToDevice);
    cudaMemcpy(dtest, test, sizeof(char), cudaMemcpyHostToDevice);

    // Set num blocks and num threads per block
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(GRID_SIZE);

    calc_histogram<<<dimGrid, dimBlock>>>(dbuff, dcount, size);

    cudaMemcpy(buff, dbuff, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(count, dcount, NUM_ELEMENTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(test, dtest, sizeof(char), cudaMemcpyDeviceToHost);

    cout << count[0] << endl;

    cudaFree(dbuff);
    cudaFree(dcount);
    free(buff);

    return rc;
}

int main (int argc, char* argv[]) {
    // Forward declarations
    int rc = 0;
    
    // Verify inputs
    rc = verify_inputs(argv);
    if (rc != 0) {
        fprintf(stderr, "Failed to verify input arguments. Appropriate "\
            "usage: ./histogram <input_matrix_file> <output_histogram_file>\n");
        app_exit(rc);
    }

    // Compute histogram math
    rc = create_histogram();
    if (rc != 0) {
        fprintf(stderr, "Failed to parse and calculate the histogram from "\
            "the input matrix\n");
        app_exit(rc);
    }

    // Exit app
    app_exit(rc);
}
