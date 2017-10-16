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

#define BLOCK_SIZE 16

// Global declaration for FILEs
FILE* file_out;
FILE* file_in;

__global__ void calc_histogram(char* element, int* zero_ptr, int* one_ptr,
    int* two_ptr, int* three_ptr, int* four_ptr, int* five_ptr, int* six_ptr,
    int* seven_ptr, int* eight_ptr, int* nine_ptr) {
    
    // Increment counter per occurances
    if (element == "0") {
        *zero_ptr += 1;
    } else if (element == "1") {
        *one_ptr += 1;
    } else if (element == "2") {
        *two_ptr += 1;
    } else if (element == "3") {
        *three_ptr += 1;
    } else if (element == "4") {
        *four_ptr += 1;
    } else if (element == "5") {
        *five_ptr += 1;
    } else if (element == "6") {
        *six_ptr += 1;
    } else if (element == "7") {
        *seven_ptr += 1;
    } else if (element == "8") {
        *eight_ptr += 1;
    } else if (element == "9") {
        *nine_ptr += 1;
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
    int rc           = 0;
    int size         = 0;
    int zero_count   = 0;
    int one_count    = 0;
    int two_count    = 0;
    int three_count  = 0;
    int four_count   = 0;
    int five_count   = 0;
    int six_count    = 0;
    int seven_count  = 0;
    int eight_count  = 0;
    int nine_count   = 0;
    char* buff;
    char* element;
    int* zero_ptr;
    int* one_ptr;
    int* two_ptr;
    int* three_ptr;
    int* four_ptr;
    int* five_ptr;
    int* six_ptr;
    int* seven_ptr;
    int* eight_ptr;
    int* nine_ptr;

    // Read the size of the file
    fseek(file_in, 0, SEEK_END);
    size = ftell(file_in);
    rewind(file_in);

    // Create heap space for buffer
    buff = reinterpret_cast<char*>(malloc((size + 1)*sizeof(char)));
    memset(buff, '\0', size);

    // Read file
    fread(buff, size, 1, file_in);

    // Malloc space for CUDA
    cudaMalloc((void**)&element, size);
    cudaMemcpy(element, buff, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&zero_ptr, 1);
    cudaMemcpy(zero_ptr, &zero_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&one_ptr, 1);
    cudaMemcpy(one_ptr, &one_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&two_ptr, 1);
    cudaMemcpy(two_ptr, &two_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&three_ptr, 1);
    cudaMemcpy(three_ptr, &three_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&four_ptr, 1);
    cudaMemcpy(four_ptr, &four_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&five_ptr, 1);
    cudaMemcpy(five_ptr, &five_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)& six_ptr, 1);
    cudaMemcpy(six_ptr, &six_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&seven_ptr, 1);
    cudaMemcpy(seven_ptr, &seven_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&eight_ptr, 1);
    cudaMemcpy(eight_ptr, &eight_count, 1, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&nine_ptr, 1);
    cudaMemcpy(nine_ptr, &nine_count, 1, cudaMemcpyHostToDevice);

    dim3 dim_block(BLOCK_SIZE, 1);
    dim3 dim_grid(1, 1);
    calc_histogram<<<dim_grid, dim_block>>>(element, zero_ptr, one_ptr,
        two_ptr, three_ptr, four_ptr, five_ptr, six_ptr, seven_ptr, eight_ptr,
        nine_ptr);

    cudaMemcpy(&zero_count, zero_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&one_count, one_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&two_count, two_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&three_count, three_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&four_count, four_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&five_count, five_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&six_count, six_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&seven_count, seven_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&eight_count, eight_ptr, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&nine_count, nine_ptr, 1, cudaMemcpyDeviceToHost);
    cudaFree(zero_ptr);
    cudaFree(one_ptr);
    cudaFree(two_ptr);
    cudaFree(three_ptr);
    cudaFree(four_ptr);
    cudaFree(five_ptr);
    cudaFree(six_ptr);
    cudaFree(seven_ptr);
    cudaFree(eight_ptr);
    cudaFree(nine_ptr);

    cout << zero_count << endl;
    cout << one_count << endl;
    cout << two_count << endl;
    cout << three_count << endl;
    cout << four_count << endl;
    cout << five_count << endl;
    cout << six_count << endl;
    cout << seven_count << endl;
    cout << eight_count << endl;
    cout << nine_count << endl;

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
