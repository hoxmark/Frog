#include "sha256.cu"
#include <cuda.h>
#include <stdio.h>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/io.h>
#include <sys/mman.h>

#define cudaCheckErrors(msg)                                                   \
    do {                                                                       \
        cudaError_t __err = cudaGetLastError();                                \
        if (__err != cudaSuccess) {                                            \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,            \
                    cudaGetErrorString(__err), __FILE__, __LINE__);            \
            fprintf(stderr, "*** FAILED - ABORTING\n");                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int num_lines = 150000000;
int line_length = 12;

char* host_passwords;
int* host_password_lengths;
int* host_start_indexes;

char* device_hashes;
int* device_password_lengths;
int* device_start_indexes;

const char* target = {"1d2adc0b54e11300dfa718b012de7f8af4befb29515b984fcf4e39bd4e96d43d"};
__constant__ unsigned char target_hex[32];

__device__ void calculate_hash(unsigned char* pass_cleartext, unsigned char* hash, int length) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, pass_cleartext, length);
    sha256_final(&ctx, hash);
}

__global__ void compare_hashes(char* hashes, int* lengths, int* start_indexes) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_length = 150000000;
    int increment = blockDim.x * gridDim.x;

    for (int i = id; i < total_length; i += increment) {
        int length = lengths[i];
        int start = start_indexes[i];

        if(start < total_length){
            unsigned char pass_cleartext[30];
            memcpy(pass_cleartext, &hashes[start], length * sizeof(char));
            pass_cleartext[length] = '\0';

            unsigned char hash[32];
            calculate_hash(pass_cleartext, hash, length);

            bool found = true;
            for (int j = 0; j < 32; j++) {
                if (hash[j] != target_hex[j]) {
                    found = false;
                    break;
                }
            }

            if (found) {
                printf("Thread %d found it! The password is %s\n", id, pass_cleartext);
            }
        }
    }
}

int main() {
    int total_length = num_lines * line_length;
    host_password_lengths = (int*)malloc(num_lines * sizeof(int));
    host_start_indexes = (int*)malloc(num_lines * sizeof(int));

    FILE* length_file = fopen("/datadrive/cracklist/password_lengths.txt", "r");
    // FILE* length_file = fopen("password_lengths.txt", "r");
    const char * file_name = "/datadrive/cracklist/passwords_one_line.txt";

    int fd = open (file_name, O_RDONLY);
    host_passwords = (char *) mmap (0, total_length, PROT_READ, MAP_PRIVATE, fd, 0);

    int i = 0;
    int counter = 0;
    int start = 0;

    // Copy all the lengths into host_password_lengths
    fscanf(length_file, "%d", &i);
    host_password_lengths[counter] = i;
    host_start_indexes[counter] = 0;
    start += i;
    counter++;
    while (!feof(length_file) && counter < num_lines) {
        fscanf(length_file, "%d", &i);
        host_password_lengths[counter] = i;
        host_start_indexes[counter] = start;
        start += i;
        counter++;
    }
    fclose(length_file);

    // Convert string hash to hex array, copy to constant memory
    const char* pos = target;
    unsigned char val[32];
    size_t count = 0;
    for (count = 0; count < sizeof(val) / sizeof(val[0]); count++) {
        sscanf(pos, "%2hhx", &val[count]);
        pos += 2;
    }
    cudaMemcpyToSymbol(target_hex, val, 32 * sizeof(unsigned char));

    cudaMalloc((void**)&device_password_lengths, num_lines * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0");
    cudaMemcpy(device_password_lengths, host_password_lengths,
               num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0");

    cudaMalloc((void**)&device_start_indexes, num_lines * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0.5");
    cudaMemcpy(device_start_indexes, host_start_indexes,
               num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0.5");

    cudaMalloc((void**)&device_hashes, total_length * sizeof(char));
    cudaCheckErrors("After cudaMalloc 1");
    cudaMemcpy(device_hashes, host_passwords, total_length * sizeof(char),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 1");

    dim3 dimGrid(1000);
    dim3 dimBlock(1000);
    double n_threads = dimGrid.x * dimBlock.x;
    printf("Each thread calculating %f hashes \n", num_lines / n_threads);

    /* Timing */
    cudaEvent_t start_time, stop;
    float elapsedTime;
    cudaEventCreate(&start_time);
    cudaEventRecord(start_time, 0);

    compare_hashes<<<dimGrid, dimBlock>>>(device_hashes, device_password_lengths, device_start_indexes);
    cudaCheckErrors("After kernel run ");
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start_time, stop);
    printf("Elapsed time: %f\n", elapsedTime);

    cudaFree(device_hashes);
    cudaCheckErrors("After free 1 ");
    cudaFree(device_start_indexes);
    cudaCheckErrors("After free 2 ");
    cudaFree(device_password_lengths);
    cudaCheckErrors("After free 3 ");

    free(host_start_indexes);
    free(host_password_lengths);

    return EXIT_SUCCESS;
}
