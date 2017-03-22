#include "sha256.cu"
#include <cuda.h>
#include <stdio.h>

#include <fcntl.h>
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

int num_passwords = 19922147;
int password_length = 30;

char* host_passwords;
int* host_password_lengths;
int* host_start_indexes;

char* device_passwords;
int* device_password_lengths;
int* device_start_indexes;
unsigned char* device_targets;

__device__ int device_num_targets;
__device__ int device_num_passwords;
__device__ size_t device_password_file_size;

__device__ void calculate_hash(unsigned char* pass_cleartext,
                               unsigned char* hash, int length) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, pass_cleartext, length);
    sha256_final(&ctx, hash);
}

__global__ void compare_hashes(char* hashes, int* lengths, int* start_indexes,
                               unsigned char* targets) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int numThreads = blockDim.x * gridDim.x;
    int num_to_calculate = device_num_passwords / numThreads;
    num_to_calculate += 1;

    if (id == 1) {
        printf("Device num: %d increment: %d \n", device_num_passwords,
               numThreads);
    }

    bool ran = false;

    int test = 0;
    int i;
    for (i = id; i < device_num_passwords; i += numThreads) {
        int length = lengths[i];
        int start = start_indexes[i];

        if ((start + length) < device_password_file_size &&
            i < device_num_passwords) {

            if (length > 15306) {
                printf("Thread id: %d Length at %d is over much %d \n", id, i,
                       length);
                length = 5;
            }
            unsigned char pass_cleartext[15306];
            memcpy(pass_cleartext, &hashes[start], length);

            pass_cleartext[length] = '\0';
            unsigned char hash[30];
            
            for (int e = 0; e < 25; e++){                
                unsigned char pass_cleartext_with_salt[15306] = "";  
                if (e<10){
                    pass_cleartext_with_salt[0] = '0';
                    pass_cleartext_with_salt[1] = (48+e); 
                } else {
                    int first_int = (e/10);
                    pass_cleartext_with_salt[0] = (48+first_int);
                    pass_cleartext_with_salt[1] = (48 + (e-(first_int*10)));                     
                }

                for (int r = 0; r < length; r++){
                        pass_cleartext_with_salt[r+2] = pass_cleartext[r];
                }

                calculate_hash(pass_cleartext_with_salt, hash, length+2);
               
                for (int k = 0; k < device_num_targets; k++) {
                    bool found = true;
                    for (int j = 0; j < 32; j++) {
                        if (hash[j] != targets[k * 32 + j]) {
                            found = false;
                            break;
                        }
                    }
                    if (found) {
                        printf("Thread %d found it! The password is %s \nsalt is: %s \n", id,
                               pass_cleartext, pass_cleartext_with_salt);
                    }
                }
            }


        } else {
            printf("Thread %d went over!: %d\n", id, i);
        }
    }
}

int main() {

    FILE* fp = fopen("targetsShort.txt", "r");
    const char* file_name =
        "/datadrive/cracklist/top100/passwords_one_line.txt";
    FILE* length_file =
        fopen("/datadrive/cracklist/top100/password_lengths.txt", "r");

    // Calculate file size
    int fd = open(file_name, O_RDONLY);
    size_t password_file_size;
    password_file_size = lseek(fd, 0, SEEK_END);
    printf("%zu \n", password_file_size);

    host_password_lengths = (int*)malloc(num_passwords * sizeof(int));
    host_start_indexes = (int*)malloc(num_passwords * sizeof(int));
    host_passwords =
        (char*)mmap(0, password_file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Copy all the lengths into host_password_lengths
    int i = 0;
    int counter = 0;
    int start = 0;
    fscanf(length_file, "%d", &i);
    host_password_lengths[counter] = i;
    host_start_indexes[counter] = 0;
    start += i;
    counter++;
    while (!feof(length_file) && counter < num_passwords) {
        fscanf(length_file, "%d", &i);
        host_password_lengths[counter] = i;
        host_start_indexes[counter] = start;
        start += i;
        counter++;
    }
    fclose(length_file);

    int host_num_targets = 0;
    int ch = 0;
    while (!feof(fp)) {
        ch = fgetc(fp);
        if (ch == '\n') {
            host_num_targets++;
        }
    }
    unsigned char* host_targets =
        (unsigned char*)malloc(32 * host_num_targets * sizeof(unsigned char));

    char* pos;
    char str[65];

    fseek(fp, 0, SEEK_SET);
    for (int i = 0; i < host_num_targets; i++) {
        if (fgets(str, 100, fp) != NULL) {
            printf("New hash: %s \n", str);
            pos = str;
            int count;
            for (count = 0; count < 32; count++) {
                sscanf(pos, "%2hhx", &host_targets[i * 32 + count]);
                pos += 2;
            }
        }
    }

    cudaMemcpyToSymbol(device_password_file_size, &password_file_size,
                       sizeof(size_t));
    cudaCheckErrors("After cudaMalloc -3");
    cudaMemcpyToSymbol(device_num_targets, &host_num_targets, sizeof(int));
    cudaCheckErrors("After cudaMalloc -2");

    cudaMemcpyToSymbol(device_num_passwords, &num_passwords, sizeof(int));
    cudaCheckErrors("After cudaMalloc -1.5");

    cudaMalloc((void**)&device_targets,
               host_num_targets * 32 * sizeof(unsigned char));
    cudaCheckErrors("After cudaMalloc -1");
    cudaMemcpy(device_targets, host_targets,
               32 * host_num_targets * sizeof(unsigned char),
               cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy -1");

    cudaMalloc((void**)&device_password_lengths, num_passwords * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0");
    cudaMemcpy(device_password_lengths, host_password_lengths,
               num_passwords * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0");

    cudaMalloc((void**)&device_start_indexes, num_passwords * sizeof(int));
    cudaCheckErrors("After cudaMalloc 0.5");
    cudaMemcpy(device_start_indexes, host_start_indexes,
               num_passwords * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 0.5");

    cudaMalloc((void**)&device_passwords, password_file_size * sizeof(char));
    cudaCheckErrors("After cudaMalloc 1");

    cudaMemcpy(device_passwords, host_passwords,
               password_file_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckErrors("After cudaMemcpy 1");

    long numGrid = num_passwords / 1024;
    numGrid += 1;
    dim3 dimGrid(32);
    dim3 dimBlock(1024);
    double n_threads = dimGrid.x * dimBlock.x;
    printf("%d threads in each grid. Each thread calculating %f hashes \n",
           numGrid, num_passwords / n_threads);

    /* Timing */
    cudaEvent_t start_time, stop;
    float elapsedTime;
    cudaEventCreate(&start_time);
    cudaEventRecord(start_time, 0);

    compare_hashes<<<dimGrid, dimBlock>>>(device_passwords,
                                          device_password_lengths,
                                          device_start_indexes, device_targets);
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
