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
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cuda.h>
#include <string>

using namespace std;

#define BLOCK_SIZE   1
#define GRID_SIZE    1
#define NUM_ELEMENTS 10 // Data is buffering weird, so giving myself some buffer room

// Global declaration for FILEs
FILE* file_out;
FILE* file_in;

__global__ void calc_histogram(char* dbuff, unsigned int* dcount, unsigned int size) {

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

__global__ void testing(char* test) {
    test[0] = 'b';
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
    unsigned int count[NUM_ELEMENTS] = { 0 };
    unsigned int* dcount;
    char tmp[1] = { 'a' };
    char* tmp_ptr;
    
    // Read the size of the file
    fseek(file_in, 0, SEEK_END);
    size = ftell(file_in) + 1;
    rewind(file_in);

    // Malloc space for CUDA
    cudaMalloc((void**)&dbuff, size*sizeof(char));
    cudaMalloc((void**)&dcount, NUM_ELEMENTS*sizeof(unsigned int));
    cudaMalloc((void**)&tmp_ptr, sizeof(char));

    // Create heap space for buffer
    buff = reinterpret_cast<char*>(malloc(size*sizeof(char)));
    memset(buff, '\0', size);

    // Read file
    fread(buff, size, sizeof(char), file_in);

    cudaMemcpy(dbuff, buff, (size*sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemcpy(dcount, count, (NUM_ELEMENTS*sizeof(unsigned int)), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_ptr, tmp, sizeof(char), cudaMemcpyHostToDevice);

    // Set num blocks and num threads per block
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(GRID_SIZE);

    calc_histogram<<<dimGrid, dimBlock>>>(dbuff, dcount, size);
    //testing<<<1, 1>>>(tmp_ptr);

    cudaMemcpy(buff, dbuff, (size*sizeof(char)), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, dcount, (NUM_ELEMENTS*sizeof(unsigned int)), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp, tmp_ptr, sizeof(char), cudaMemcpyHostToDevice);

    // Write to file
    fprintf(file_out, "0 => %u\n", count[0]);
    fprintf(file_out, "1 => %u\n", count[1]);
    fprintf(file_out, "2 => %u\n", count[2]);
    fprintf(file_out, "3 => %u\n", count[3]);
    fprintf(file_out, "4 => %u\n", count[4]);
    fprintf(file_out, "5 => %u\n", count[5]);
    fprintf(file_out, "6 => %u\n", count[6]);
    fprintf(file_out, "7 => %u\n", count[7]);
    fprintf(file_out, "8 => %u\n", count[8]);
    fprintf(file_out, "9 => %u\n", count[9]);

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
