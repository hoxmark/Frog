#include <stdio.h>
#include <cuda.h>
#include "sha256.cu"

#define cudaCheckErrors(msg)                                   \
    do                                                         \
    {                                                          \
        cudaError_t __err = cudaGetLastError();                \
        if (__err != cudaSuccess)                              \
        {                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err),            \
                    __FILE__, __LINE__);                       \
            fprintf(stderr, "*** FAILED - ABORTING\n");        \
            exit(1);                                           \
        }                                                      \
    } while (0)

int num_lines = 1000000;
int line_length = 64;

char *host_hashes;
int *host_password_lengths;
int *host_start_indexes;

char *device_hashes;
int *device_password_lengths;
int *device_start_indexes;


const char *target = {"7e7e5e4ff50373b062278d1be961570afbb288a588e590ecc07648818389e32e"};
__constant__ unsigned char target_hex[32];
__device__ int d_answer = 0;
__device__ int hash_length = 64;

__global__ void compare_hashes(char *hashes, int *lengths, int *start_indexes)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int length = lengths[id];
    int start = start_indexes[id];

    unsigned char pass_cleartext[30];
    memcpy(pass_cleartext, &hashes[start], length * sizeof(char));
    pass_cleartext[length] = '\0';

    unsigned char hash[32];

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, pass_cleartext, length);
    sha256_final(&ctx,hash);

    bool found = true;
    for(int i = 0; i<32; i++){
        if(hash[i] != target_hex[i]){
            found = false;
            break;
        }
    }
    // bool isEqual = memcmp(hash, target_hex, 32);

    if(found){
        printf("Thread %d found it! The password is %s\n", id, pass_cleartext);
    }
}

int main()
{
    int total_length = num_lines * line_length;
    host_password_lengths = (int *)malloc(num_lines * sizeof(int));
    host_start_indexes = (int *)malloc(num_lines * sizeof(int));
    host_hashes = (char *)malloc(total_length * sizeof(char));

    // Copy all the passwords into host_hashes
    FILE *fp = fopen("passwords_one_line.txt", "r");
    if (fgets(host_hashes, total_length, fp) == NULL)
    {
        printf("We failed\n");
    }

    // Copy all the lengths into host_password_lengths
    int i = 0;
    int counter = 0;
    int start = 0;

    FILE* length_file = fopen ("password_lengths.txt", "r");
    fscanf (length_file, "%d", &i);
    host_password_lengths[counter] = i;
    host_start_indexes[counter] = 0;
    start += i;
    counter++;
    while (!feof (length_file))
    {
        fscanf (length_file, "%d", &i);
        host_password_lengths[counter] = i;
        host_start_indexes[counter] = start;
        start += i;
        counter++;
    }
    fclose (length_file);

    // Convert string hash to hex array, copy to constant memory
    const char *pos = target;
    unsigned char val[32];
    size_t count = 0;
    for(count = 0; count < sizeof(val)/sizeof(val[0]); count++) {
        sscanf(pos, "%2hhx", &val[count]);
        pos += 2;
    }
    cudaMemcpyToSymbol(target_hex, val, 32*sizeof(unsigned char));

    cudaMalloc((void **) &device_password_lengths, num_lines * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0");
    cudaMemcpy(device_password_lengths, host_password_lengths, num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0");

    cudaMalloc((void **) &device_start_indexes, num_lines * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0.5");
    cudaMemcpy(device_start_indexes, host_start_indexes, num_lines * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0.5");

    cudaMalloc((void **)&device_hashes, total_length * sizeof(char));
    cudaCheckErrors("After cudaMalloc 1");
    cudaMemcpy(device_hashes, host_hashes, total_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 1");

    dim3 dimGrid(1000);
    dim3 dimBlock(1000);

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
    cudaFree(device_start_indexes);
    cudaFree(device_password_lengths);

    free(host_hashes);
    free(host_start_indexes);
    free(host_password_lengths);

    cudaCheckErrors("After free 1 ");

    return EXIT_SUCCESS;
}
