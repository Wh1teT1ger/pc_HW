/*
 * md5.cuh CUDA Implementation of MD5 digest       
 *
 * Date: 12 June 2019
 * Revision: 1
 * 
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 */

/*************************** HEADER FILES ***************************/
#pragma once
#include "config.h"
#include <cuda_runtime.h>

#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define MD5_BLOCK_SIZE 16               // MD5 outputs a 16 byte digest

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[4];
} CUDA_MD5_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) ((a << b) | (a >> (32-b)))
#endif

#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & z) | (y & ~z))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | ~z))

#define FF(a,b,c,d,m,s,t) { a += F(b,c,d) + m + t; \
                            a = b + ROTLEFT(a,s); }
#define GG(a,b,c,d,m,s,t) { a += G(b,c,d) + m + t; \
                            a = b + ROTLEFT(a,s); }
#define HH(a,b,c,d,m,s,t) { a += H(b,c,d) + m + t; \
                            a = b + ROTLEFT(a,s); }
#define II(a,b,c,d,m,s,t) { a += I(b,c,d) + m + t; \
                            a = b + ROTLEFT(a,s); }

/*********************** FUNCTION DEFINITIONS ***********************/

__device__ void cuda_md5_transform(CUDA_MD5_CTX *ctx, const BYTE data[]);

__device__ void cuda_md5_init(CUDA_MD5_CTX *ctx);

__device__ void cuda_md5_update(CUDA_MD5_CTX *ctx, const BYTE data[], size_t len);

__device__ void cuda_md5_final(CUDA_MD5_CTX *ctx, BYTE hash[]);

__device__ void compute_md5(const unsigned char* input, size_t input_len, unsigned char* output);

