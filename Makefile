CC=gcc
CFLAGS=-Wall -lssl -lcrypto -L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include

test: test.c
	$(CC) -o test test.c $(CFLAGS)
