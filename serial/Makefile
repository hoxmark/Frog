CC=gcc
CFLAGS=-Wall -lssl -lcrypto -L/usr/local/opt/openssl/lib -I/usr/local/opt/openssl/include

openssl_md5: openssl_md5.c
	$(CC) -o openssl_md5 openssl_md5.c $(CFLAGS)

make crack_md5: crack_md5.c
	$(CC) -o crack_md5 crack_md5.c $(CFLAGS)

make crack_sha1: crack_sha1.c
	$(CC) -o crack_sha1 crack_sha1.c $(CFLAGS)

make crack_sha256: crack_sha256.c
	$(CC) -o crack_sha256 crack_sha256.c $(CFLAGS)
