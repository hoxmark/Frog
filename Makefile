NVCC=/usr/local/cuda-8.0/bin/nvcc
CC=gcc
CFLAGS=-Wall -lssl -lcrypto -L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include

test: test.c
	$(CC) -o ./output/test test.c $(CFLAGS)

crack: crack.cu
	$(NVCC) -O3 crack.cu -o crack $(CFLAGS)

test_buffer: test_buffer.c
	$(CC) -o ./output/test_buffer test_buffer.c $(CFLAGS)
