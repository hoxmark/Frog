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
char *device_hashes;
__device__ char *target = "93eb2df432f7d1b7281568260bbf03e06bc0b5b344ea41a1bd2ac440f5655a0f";
__device__ int d_answer = 0;

__global__ void compare_hashes(char *hashes)
{
    // int id = threadIdx.x + blockIdx.x * blockDim.x;
    // bool found = true;
    // int i;
    // char hash[64];
    // for (i = id * 64; i < (id + 1) * 64; i++)
    // {
    //     hash[i - id * 64] = hashes[i];
    //     if (hashes[i] != target[i - id * 64])
    //     {
    //         found = false;
    //         break;
    //     }
    // }
    // // printf("%d: \t %s \n", id, hash);
    // if (found)
    // {
    //     printf("We found it %d \n", id);
    //     d_answer = 1;
    // }

    unsigned char text1[]={"abc"}, hash[32];

    int idx;
    SHA256_CTX ctx;

   // Hash one
   sha256_init(&ctx);
   sha256_update(&ctx,text1, 3);
   sha256_final(&ctx,hash);
   print_hash(hash);
}

int main()
{
    int total_length = num_lines * line_length;
    FILE *fp = fopen("/datadrive/crackstation_hash.txt", "r");
    host_hashes = (char *)malloc(num_lines * line_length * sizeof(char));
    if (fgets(host_hashes, total_length, fp) == NULL)
    {
        printf("We failed\n");
    }
    cudaMalloc((void **)&device_hashes, total_length * sizeof(char));
    cudaCheckErrors("After cudaMalloc 1");
    cudaMemcpy(device_hashes, host_hashes, total_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 1");


    cudaCheckErrors("Before kernel run ");
    dim3 dimBlock(1, 1);
    dim3 dimGrid(1, 1);

    /* Timing */
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    compare_hashes<<<dimGrid, dimBlock>>>(device_hashes);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f\n", elapsedTime);
    cudaCheckErrors("After kernel run ");

    cudaCheckErrors("After free 1 ");

    return EXIT_SUCCESS;
}
