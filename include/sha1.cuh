/*
 * sha1.cuh CUDA Implementation of SHA1 Hashing    
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

 
#pragma once
#include "config.h"
#include <cuda_runtime.h>
/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>

/****************************** MACROS ******************************/
#define SHA1_BLOCK_SIZE 20              // SHA1 outputs a 20 byte digest

/**************************** DATA TYPES ****************************/
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[5];
	WORD k[4];
} CUDA_SHA1_CTX;

/****************************** MACROS ******************************/
#ifndef ROTLEFT
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#endif

__device__  __forceinline__ void cuda_sha1_transform(CUDA_SHA1_CTX *ctx, const BYTE data[]);

__device__ void cuda_sha1_init(CUDA_SHA1_CTX *ctx);

__device__ void cuda_sha1_update(CUDA_SHA1_CTX *ctx, const BYTE data[], size_t len);

__device__ void cuda_sha1_final(CUDA_SHA1_CTX *ctx, BYTE hash[]);

__device__ void compute_sha1(const unsigned char* input, size_t input_len, unsigned char* output);
