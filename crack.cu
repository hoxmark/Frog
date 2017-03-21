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

long num_lines = 440000000;
int line_length = 9;

char* host_passwords;
int* host_password_lengths;
int* host_start_indexes;

char* device_passwords;
int* device_password_lengths;
int* device_start_indexes;
unsigned char *device_targets;

__constant__ unsigned char target_hex[32];

__device__ void calculate_hash(unsigned char* pass_cleartext, unsigned char* hash, int length) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, pass_cleartext, length);
    sha256_final(&ctx, hash);
}

__global__ void compare_hashes(char* hashes, int* lengths, int* start_indexes, unsigned char *targets) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    long num_lines = 440000000;
    int numThreads = blockDim.x * gridDim.x;

    int num_to_calculate = num_lines / numThreads;
    num_to_calculate += 1;

    if(id == 1){
        printf("Increment: %d num_to_calculate: %d\n", numThreads, num_to_calculate);
    }

    long i;
    // for (i = id; i < num_lines; i += numThreads) {
    for(i = 0; i < num_to_calculate; i++) {
        int index = id + i * numThreads;

        if (index < num_lines){
            int length = lengths[index];
            int start = start_indexes[index];

            if (start < num_lines) {
                unsigned char pass_cleartext[30];
                memcpy(pass_cleartext, &hashes[start], length * sizeof(char));
                pass_cleartext[length] = '\0';

                unsigned char hash[32];
                calculate_hash(pass_cleartext, hash, length);

                for(int i = 0; i < 3; i++){
                    bool found = true;
                    for (int j = 0; j < 32; j++) {
                        if (hash[j] != targets[i * 32 + j]) {
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
    }
}


int main() {

    FILE *fp = fopen("targets.txt", "r");
    const char * file_name = "/datadrive/cracklist/hashesorg/passwords_one_line.txt";
    FILE* length_file = fopen("/datadrive/cracklist/hashesorg/password_lengths.txt", "r");

    // Calculate file size
    int fd = open (file_name, O_RDONLY);
    size_t password_file_size;
    password_file_size = lseek(fd, 0, SEEK_END);

    host_password_lengths = (int*)malloc(num_lines * sizeof(int));
    host_start_indexes = (int*)malloc(num_lines * sizeof(int));
    host_passwords = (char *) mmap (0, password_file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Copy all the lengths into host_password_lengths
    int i = 0;
    int counter = 0;
    int start = 0;
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
    int target_count = 3;
    unsigned char *val = (unsigned char *)malloc(32 * target_count * sizeof(unsigned char));

    char *pos;
    char str[65];
    for(int i = 0; i < target_count; i++){
        if( fgets (str, 100, fp) != NULL ) {
            printf("New hash: %s \n", str);
            pos = str;
            int count;
            for (count = 0; count < 32; count++) {
                sscanf(pos, "%2hhx", &val[i * 32 + count]);
                pos += 2;
            }
        }
    }
    
    cudaMalloc((void **) &device_targets, target_count * 32 * sizeof(unsigned char));
    cudaCheckErrors("After cudaMalloc -1");
    cudaMemcpy(device_targets, val, 32 * target_count * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy -1");

    cudaMalloc((void**)&device_password_lengths, num_lines * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0");
    cudaMemcpy(device_password_lengths, host_password_lengths, num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0");

    cudaMalloc((void**)&device_start_indexes, num_lines * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0.5");
    cudaMemcpy(device_start_indexes, host_start_indexes, num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0.5");

    cudaMalloc((void**) &device_passwords, password_file_size * sizeof(char));
    cudaCheckErrors("After cudaMalloc 1");

    cudaMemcpy(device_passwords, host_passwords, password_file_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 1");

    long numGrid = num_lines / 1024;
    numGrid += 1;
    dim3 dimGrid(1000);
    dim3 dimBlock(1000);
    double n_threads = dimGrid.x * dimBlock.x;
    printf("Grid size: %d. Each thread calculating %f hashes \n", numGrid, num_lines / n_threads);

    /* Timing */
    cudaEvent_t start_time, stop;
    float elapsedTime;
    cudaEventCreate(&start_time);
    cudaEventRecord(start_time, 0);

    compare_hashes<<<dimGrid, dimBlock>>>(device_passwords, device_password_lengths, device_start_indexes, device_targets);
    cudaCheckErrors("After kernel run ");
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start_time, stop);
    printf("Elapsed time: %f\n", elapsedTime);

    cudaFree(device_passwords);
    cudaCheckErrors("After free 1 ");
    cudaFree(device_start_indexes);
    cudaCheckErrors("After free 2 ");
    cudaFree(device_password_lengths);
    cudaCheckErrors("After free 3 ");

    free(host_start_indexes);
    free(host_password_lengths);

    return EXIT_SUCCESS;
}
