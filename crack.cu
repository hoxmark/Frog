
#include "md5.cu"
#include "sha1.cu"
#include "sha256.cu"
#include <cuda.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/io.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

/*-------------
* CONFIGURATION
* ----TYPES---
* -- 1: MD5 --
* -- 2: SHA1 -
* -- 3: SHA256
*-------------*/
#define type 1

#define password_length 30
#if type == 1
#define hash_length 16
#endif
#if type == 2
#define hash_length 20
#endif
#if type == 3
#define hash_length 32
#endif

//  Number of lines in the dictionary file. Must correspond with what is found
//  in dictionary_file_name

// Hashesorg
// #define num_passwords 446426204

// merged
#define num_passwords 19922147

// top 100
// #define num_passwords 999999

char* host_dictionary_file;
int* host_dictionary_word_lengths;
int* host_start_indexes;
unsigned char* host_targets;
int* host_num_cracked;

char* device_passwords;
int* device_dictionary_word_lengths;
int* device_start_indexes;
unsigned char* device_targets;
int* device_num_cracked;

__constant__ int device_num_targets;
__constant__ size_t device_password_file_size;

__device__ void print_hash(unsigned char hash[]) {
    int idx;
    for (idx = 0; idx < 32; idx++)
        printf("%02x", hash[idx]);
    printf("\n");
}

__device__ void calculate_hash(unsigned char* pass_cleartext,
                               unsigned char* hash, int length) {

    if (type == 1) {
        MD5_CTX ctx;
        md5_init(&ctx);
        md5_update(&ctx, pass_cleartext, length);
        md5_final(&ctx, hash);
    }

    if (type == 2) {
        SHA1_CTX ctx;
        sha1_init(&ctx);
        sha1_update(&ctx, pass_cleartext, length);
        sha1_final(&ctx, hash);
    }

    if (type == 3) {
        SHA256_CTX ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, pass_cleartext, length);
        sha256_final(&ctx, hash);
    }
}

__global__ void compare_hashes(char* hashes, int* lengths, int* start_indexes,
                               unsigned char* targets, int* num_cracked) {
    extern __shared__ int sdata[];

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int numThreads = blockDim.x * gridDim.x;
    int num_to_calculate = num_passwords / numThreads;
    num_to_calculate += 1;

    int cracked = 0;
    int i, k, j;
    // unsigned char target[hash_length];
    unsigned char hash[hash_length];
    for (i = id; i < num_passwords; i += numThreads) {
        int length = lengths[i];
        int start = start_indexes[i];

        if (start < device_password_file_size) {
            unsigned char pass_cleartext[password_length];
            if (length < password_length) {
                memcpy(pass_cleartext, &hashes[start], length);
                pass_cleartext[length] = '\0';

                calculate_hash(pass_cleartext, hash, length);

                for (k = 0; k < device_num_targets; k++) {
                    bool found = true;
                    for (j = 0; j < hash_length; j++) {
                        if (hash[j] != targets[k * hash_length + j]) {
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        cracked++;
                        printf(" %s \n", pass_cleartext);
                    }
                }
            }
        }
    }

    // Do reduction sum to get the counts
    unsigned int tid = threadIdx.x;
    sdata[tid] = cracked;

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        num_cracked[blockIdx.x] = sdata[0];
}

int main() {
    int i = 0;
    int num_threads = 1024;
    int num_blocks = 32;

    FILE* fp;
    if (type == 1) {
        fp = fopen("passwords/eharmony.txt", "r");
    }

    if (type == 2) {
        fp = fopen("passwords/unmasked.lst", "r");
    }

    if (type == 3) {
        fp = fopen("passwords/targets.txt", "r");
    }

    const char* dictionary_file_name =
        "/datadrive/cracklist/merged/passwords_one_line.txt";
    FILE* length_file =
        fopen("/datadrive/cracklist/merged/password_lengths.txt", "r");

    // Calculate file size
    int fd = open(dictionary_file_name, O_RDONLY);
    size_t password_file_size;
    password_file_size = lseek(fd, 0, SEEK_END);
    printf("Total file size of the wordlist: %zu \n", password_file_size);

    // Allocate space + read the word lengths and start indexes
    host_dictionary_word_lengths = (int*)malloc(num_passwords * sizeof(int));
    host_start_indexes = (int*)malloc(num_passwords * sizeof(int));
    // Memory map the dictionary file
    host_dictionary_file =
        (char*)mmap(0, password_file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Copy all the lengths into host_dictionary_word_lengths
    i = 0;
    int counter = 0;
    int start = 0;
    fscanf(length_file, "%d", &i);
    host_dictionary_word_lengths[counter] = i;
    host_start_indexes[counter] = 0;
    start += i;
    counter++;
    while (!feof(length_file) && counter < num_passwords) {
        fscanf(length_file, "%d", &i);
        host_dictionary_word_lengths[counter] = i;
        host_start_indexes[counter] = start;
        start += i;
        counter++;
    }
    fclose(length_file);

    // Count the number of targets
    int host_num_targets = 0;
    int ch = 0;
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') {
            host_num_targets++;
        }
    }

    // Hardocde targets for testing purposes
    host_num_targets = 10000;
    host_targets = (unsigned char*)malloc(hash_length * host_num_targets *
                                          sizeof(unsigned char));

    // Read the targets into host_targets
    char* pos;
    char str[65];
    fseek(fp, 0, SEEK_SET);
    for (i = 0; i < host_num_targets; i++) {
        if (fgets(str, 100, fp) != NULL) {
            pos = str;
            int count;
            for (count = 0; count < hash_length; count++) {
                sscanf(pos, "%2hhx", &host_targets[i * hash_length + count]);
                pos += 2;
            }
        }
    }

    // Array that will hold the number of cracked passwords per block
    host_num_cracked = (int*)malloc(num_blocks * sizeof(int));
    host_num_targets = 10000;

    cudaMalloc((void**)&device_num_cracked, num_blocks * sizeof(int));
    cudaCheckErrors("After cudamalloc -2");

    // Copy constants to constant memory
    cudaMemcpyToSymbol(device_password_file_size, &password_file_size,
                       sizeof(size_t));
    cudaCheckErrors("After cudaMemcpyToSymbol 1");
    cudaMemcpyToSymbol(device_num_targets, &host_num_targets, sizeof(int));
    cudaCheckErrors("After cudaMemcpyToSymbol 2");

    // Allocate and copy the rest of the data to device global memory
    cudaMalloc((void**)&device_targets,
               host_num_targets * hash_length * sizeof(unsigned char));
    cudaCheckErrors("After cudaMalloc -1");
    cudaMemcpy(device_targets, host_targets,
               hash_length * host_num_targets * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy -1");

    cudaMalloc((void**)&device_dictionary_word_lengths,
               num_passwords * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0");
    cudaMemcpy(device_dictionary_word_lengths, host_dictionary_word_lengths,
               num_passwords * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0");

    cudaMalloc((void**)&device_start_indexes, num_passwords * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0.5");
    cudaMemcpy(device_start_indexes, host_start_indexes,
               num_passwords * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0.5");

    cudaMalloc((void**)&device_passwords, password_file_size * sizeof(char));
    cudaCheckErrors("After cudaMalloc 1");
    cudaMemcpy(device_passwords, host_dictionary_file,
               password_file_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 1");

    long numGrid = num_passwords / num_threads;
    numGrid += 1;
    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_threads);
    double n_threads = dimGrid.x * dimBlock.x;
    printf("%ld threads in each grid. Each thread calculating %f hashes \n",
           numGrid, num_passwords / n_threads);

    /* Timing */
    cudaEvent_t start_time, stop;
    float elapsedTime;
    cudaEventCreate(&start_time);
    cudaEventRecord(start_time, 0);

    printf("Starting computation \n");
    compare_hashes<<<dimGrid, dimBlock, num_threads * sizeof(int)>>>(
        device_passwords, device_dictionary_word_lengths, device_start_indexes,
        device_targets, device_num_cracked);
    cudaCheckErrors("After kernel run ");
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start_time, stop);
    printf("Elapsed time in CUDA kernel: %f\n", elapsedTime);
    cudaCheckErrors("After memcpy +2");
    cudaMemcpy(host_num_cracked, device_num_cracked, sizeof(int) * num_blocks,
               cudaMemcpyDeviceToHost);
    cudaCheckErrors("After memcpy +3");

    // Count how many passwords we cracked
    int count = 0;
    for (i = 0; i < num_blocks; i++) {
        count += host_num_cracked[i];
    }
    printf("We cracked %d passwords \n", count);

    cudaFree(device_passwords);
    cudaCheckErrors("After free 1 ");
    cudaFree(device_start_indexes);
    cudaCheckErrors("After free 2 ");
    cudaFree(device_dictionary_word_lengths);
    cudaCheckErrors("After free 3 ");

    // Can't free mmapped() file for some reason
    // free(host_dictionary_file);
    free(host_dictionary_word_lengths);
    free(host_start_indexes);

    return EXIT_SUCCESS;
}
