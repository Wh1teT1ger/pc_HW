#include <iostream>
#include <cuda_runtime.h>
#include <md5.cuh>
#include <sha1.cuh>
#include <sha256.cuh>
#include <cstring>
#include <cstdio>

#define SHA1_BLOCK_SIZE 20 
#define MD5_BLOCK_SIZE 16
#define SHA256_BLOCK_SIZE 32

// Helper function to convert binary hash to hexadecimal string
__device__ void to_hex(const unsigned char* hash, size_t length, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    for (size_t i = 0; i < length; ++i) {
        hex_str[i * 2] = hex_chars[(hash[i] >> 4) & 0xF];  // High nibble
        hex_str[i * 2 + 1] = hex_chars[hash[i] & 0xF];     // Low nibble
    }
    hex_str[length * 2] = '\0';
}


// Compute nested hashes
__global__ void compute_nested_hash_kernel(const char* input, int current_input_len, const int* hash_sequence, int num_hashes,
                                           char* result) {
    unsigned char intermediate_hash[SHA256_BLOCK_SIZE];  // Largest buffer size for SHA-256
    unsigned char current_input[SHA256_BLOCK_SIZE];
    //int current_input_len = strlen(input);

    // Copy input to the current buffer
    memcpy(current_input, input, current_input_len);

    // Perform nested hashing based on the sequence
    for (int i = 0; i < num_hashes; ++i) {
        if (hash_sequence[i] == 0) {
            compute_md5(current_input, current_input_len, intermediate_hash);
            current_input_len = MD5_BLOCK_SIZE;
        } else if (hash_sequence[i] == 1) {
            compute_sha1(current_input, current_input_len, intermediate_hash);
            current_input_len = SHA1_BLOCK_SIZE;
        } else if (hash_sequence[i] == 2) {
            compute_sha256(current_input, current_input_len, intermediate_hash);
            current_input_len = SHA256_BLOCK_SIZE;
        }
        memcpy(current_input, intermediate_hash, current_input_len);
    }

    // Convert the final hash to hexadecimal and store in result
    to_hex(intermediate_hash, current_input_len, result);
}

int main() {
    const char* word = "example";
    const int hash_sequence[] = {2, 2, 2};  // MD5 → SHA-1 → SHA-256
    int num_hashes = sizeof(hash_sequence) / sizeof(hash_sequence[0]);

    // Allocate memory on the GPU
    char* d_word;
    int* d_hash_sequence;
    char* d_result;
    cudaMalloc(&d_word, strlen(word) + 1);
    cudaMalloc(&d_hash_sequence, num_hashes * sizeof(int));
    cudaMalloc(&d_result, (SHA256_BLOCK_SIZE * 2 + 1) * sizeof(char));

    cudaMemcpy(d_word, word, strlen(word) + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_sequence, hash_sequence, num_hashes * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    compute_nested_hash_kernel<<<1, 1>>>(d_word, strlen(word), d_hash_sequence, num_hashes, d_result);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy the result back to the host
    char result[SHA256_BLOCK_SIZE * 2 + 1];
    cudaMemcpy(result, d_result, (SHA256_BLOCK_SIZE * 2 + 1) * sizeof(char), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Word: " << word << std::endl;
    std::cout << "Nested Hash: " << result << std::endl;

    // Free GPU memory
    cudaFree(d_word);
    cudaFree(d_hash_sequence);
    cudaFree(d_result);

    return 0;
}