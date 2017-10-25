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
#define THREAD_SIZE  16
#define NUM_ELEMENTS 10 

// Global declaration for FILEs
FILE* file_out;
FILE* file_in;

__global__ void calc_histogram(char* dbuff, unsigned int* dcount, unsigned int size, float stride) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int start_pos = stride * index;
    unsigned int stop_pos = start_pos + stride;
    unsigned int lcount[10] = { 0 };

    if (size < stop_pos) {
        stop_pos = size;
    }

    for (unsigned int i = start_pos; i < stop_pos; i++) {
        // Increment counter per occurances
        if (dbuff[i] == '0') {
            lcount[0] += 1;
        } else if (dbuff[i] == '1') {
            lcount[1] += 1;
        } else if (dbuff[i] == '2') {
            lcount[2] += 1;
        } else if (dbuff[i] == '3') {
            lcount[3] += 1;
        } else if (dbuff[i] == '4') {
            lcount[4] += 1;
        } else if (dbuff[i] == '5') {
            lcount[5] += 1;
        } else if (dbuff[i] == '6') {
            lcount[6] += 1;
        } else if (dbuff[i] == '7') {
            lcount[7] += 1;
        } else if (dbuff[i] == '8') {
            lcount[8] += 1;
        } else if (dbuff[i] == '9') {
            lcount[9] += 1;
        }
    }

    dcount[0] += lcount[0];
    dcount[1] += lcount[1];
    dcount[2] += lcount[2];
    dcount[3] += lcount[3];
    dcount[4] += lcount[4];
    dcount[5] += lcount[5];
    dcount[6] += lcount[6];
    dcount[7] += lcount[7];
    dcount[8] += lcount[8];
    dcount[9] += lcount[9];
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
    float stride;
    
    // Read the size of the file
    fseek(file_in, 0, SEEK_END);
    size = ftell(file_in) + 1;
    rewind(file_in);

    // Malloc space for CUDA
    cudaMalloc((void**)&dbuff, size*sizeof(char));
    cudaMalloc((void**)&dcount, NUM_ELEMENTS*sizeof(unsigned int));

    // Create heap space for buffer
    buff = reinterpret_cast<char*>(malloc(size*sizeof(char)));
    memset(buff, '\0', size);

    // Read file
    fread(buff, size, sizeof(char), file_in);

    cudaMemcpy(dbuff, buff, (size*sizeof(char)), cudaMemcpyHostToDevice);
    cudaMemcpy(dcount, count, (NUM_ELEMENTS*sizeof(unsigned int)), cudaMemcpyHostToDevice);

    stride = ceil(float(size / (BLOCK_SIZE * THREAD_SIZE)));

    calc_histogram<<<THREAD_SIZE, BLOCK_SIZE>>>(dbuff, dcount, size, stride);

    cudaMemcpy(buff, dbuff, (size*sizeof(char)), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, dcount, (NUM_ELEMENTS*sizeof(unsigned int)), cudaMemcpyDeviceToHost);

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
