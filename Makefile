NVCC=/usr/local/cuda-8.0/bin/nvcc
CC=gcc
CFLAGS=-Wall -lssl -lcrypto -L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include

test: test.c
	$(CC) -o ./output/test test.c $(CFLAGS)

crack: crack.cu
	$(NVCC) -arch=compute_30 crack.cu -o crack

crackb: crackb.cu
	$(NVCC) -arch=compute_30 crackb.cu -o crackb

tcrack: tcrack.cu
	$(NVCC) -arch=compute_30 tcrack.cu -o crack

main: main.cu
	$(NVCC) -arch=compute_30 main.cu -o main

test_buffer: test_buffer.c
	$(CC) -o ./output/test_buffer test_buffer.c $(CFLAGS)

cracktest: cracktest.cu
	$(NVCC) -arch=compute_30 cracktest.cu -o cracktest

comparesha: comparesha.c
	$(CC) -o ./output/comparesha comparesha.c $(CFLAGS)

crack_md5: crack_md5.cu
	$(NVCC) -arch=compute_30 crack_md5.cu -o crack_md5

crack_sha1: crack_sha1.cu
	$(NVCC) -arch=compute_30 crack_sha1.cu -o crack_sha1
