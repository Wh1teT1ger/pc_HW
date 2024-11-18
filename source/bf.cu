#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <argparse/argparse.hpp>
#include <sha256.cuh>
#include <sha1.cuh>
#include <md5.cuh>
#include <chrono>

// Convert hex string to byte array
bool hex_to_bytes(const std::string& hex, unsigned char* bytes, int expected_length) {
    if (hex.length() != expected_length * 2) {
        return false;
    }
    for (size_t i = 0; i < expected_length; ++i) {
        std::string byte_str = hex.substr(i * 2, 2);
        bytes[i] = static_cast<unsigned char>(std::stoi(byte_str, nullptr, 16));
    }
    return true;
}

// CUDA Kernel for nested hashing
__global__ void nested_hash_kernel(char* charset, int charset_length, int max_len,
                                   unsigned long long start_idx, unsigned long long total_candidates,
                                   unsigned char* target, bool* found, char* result, int* hash_types, int hash_count) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;

    if (idx >= total_candidates || *found) {
        return;
    }

    char candidate[16];
    unsigned long long current_idx = idx;
    for (int i = 0; i < max_len; ++i) {
        candidate[i] = charset[current_idx % charset_length];
        current_idx /= charset_length;
    }

    unsigned char intermediate[32]; // Buffer for intermediate hash results
    unsigned char md5_hash[16];
    unsigned char sha1_hash[20];
    unsigned char sha256_hash[32];

    // Initialize with the candidate
    memcpy(intermediate, candidate, max_len);

    // Apply hash functions in sequence
    for (int i = 0; i < hash_count; ++i) {
        switch (hash_types[i]) {
            case 0: // MD5
                compute_md5(intermediate, max_len, md5_hash);
                memcpy(intermediate, md5_hash, 16);
                max_len = 16;
                break;
            case 1: // SHA1
                compute_sha1(intermediate, max_len, sha1_hash);
                memcpy(intermediate, sha1_hash, 20);
                max_len = 20;
                break;
            case 2: // SHA256
                compute_sha256(intermediate, max_len, sha256_hash);
                memcpy(intermediate, sha256_hash, 32);
                max_len = 32;
                break;
        }
    }

    // Compare with the target hash
    bool match = true;
    for (int i = 0; i < 32; ++i) {
        if (intermediate[i] != target[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        *found = true;
        for (int i = 0; i < max_len; ++i) {
            result[i] = candidate[i];
        }
    }
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("brute_force_nested_hashes");

    program.add_argument("target_hash")
        .help("The target hash to brute force");

    program.add_argument("hash_sequence")
        .help("Comma-separated hash sequence (e.g., MD5,SHA1,SHA256)")
        .required();


    program.add_argument("--charset")
        .help("Character set to use for generating combinations (default: abcdef)")
        .default_value(std::string("abcdefghijklmnopqrstuvwxyz"));

    program.add_argument("--max-length")
        .help("Maximum length of combinations (default: 8)")
        .scan<'i', int>()
        .default_value(8);


    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Parse arguments
    int max_len = program.get<int>("--max-length");
    std::string target_hash_hex = program.get<std::string>("target_hash");
    std::string hash_sequence_str = program.get<std::string>("hash_sequence");
    std::string charset_str = program.get<std::string>("--charset");

    if (max_len > 16) {
        std::cerr << "Error: Max length cannot exceed 16." << std::endl;
        return 1;
    }

    // Convert target hash from hex to bytes
    unsigned char target_hash[32];
    if (!hex_to_bytes(target_hash_hex, target_hash, 32)) {
        std::cerr << "Error: Invalid target hash format." << std::endl;
        return 1;
    }

    // Parse hash sequence
    std::vector<std::string> hash_sequence;
    std::stringstream ss(hash_sequence_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        hash_sequence.push_back(token);
    }

    int hash_count = hash_sequence.size();
    int hash_types[3]; // Support up to 3 hash functions
    for (int i = 0; i < hash_count; ++i) {
        if (hash_sequence[i] == "MD5") hash_types[i] = 0;
        else if (hash_sequence[i] == "SHA1") hash_types[i] = 1;
        else if (hash_sequence[i] == "SHA256") hash_types[i] = 2;
        else {
            std::cerr << "Error: Unsupported hash type: " << hash_sequence[i] << std::endl;
            return 1;
        }
    }

    int charset_length = charset_str.length();
    unsigned long long total_candidates = pow(charset_length, max_len);

    // CUDA-specific variables
    const int threads_per_block = 256;
    const unsigned long long chunk_size = 2L << 25;
    const unsigned long long num_chunks = (total_candidates + chunk_size - 1) / chunk_size;

    char* d_charset;
    unsigned char* d_target;
    bool* d_found;
    char* d_result;
    int* d_hash_types;

    bool found = false;
    char result[16] = {0};

    cudaMalloc(&d_charset, charset_length * sizeof(char));
    cudaMalloc(&d_target, 32 * sizeof(unsigned char));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_result, max_len * sizeof(char));
    cudaMalloc(&d_hash_types, hash_count * sizeof(int));

    cudaMemcpy(d_charset, charset_str.c_str(), charset_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_hash, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &found, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_types, hash_types, hash_count * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Start" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned long long chunk = 0; chunk < num_chunks; ++chunk) {
        std::cout << "Start chunk " << chunk << std::endl;
        unsigned long long start_idx = chunk * chunk_size;
        unsigned long long end_idx = std::min(start_idx + chunk_size, total_candidates);
        unsigned long long candidates_in_chunk = end_idx - start_idx;

        unsigned long long blocks_per_grid = (candidates_in_chunk + threads_per_block - 1) / threads_per_block;

        nested_hash_kernel<<<blocks_per_grid, threads_per_block>>>(d_charset, charset_length, max_len,
                                                                   start_idx, total_candidates, d_target, d_found, d_result, d_hash_types, hash_count);
        cudaDeviceSynchronize();

        cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        if (found) {
            cudaMemcpy(result, d_result, max_len * sizeof(char), cudaMemcpyDeviceToHost);
            break;
        }
    }
    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (found) {
        std::cout << "Match found: " << result << std::endl;
    } else {
        std::cout << "No match found." << std::endl;
    }

    cudaFree(d_charset);
    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_result);
    cudaFree(d_hash_types);

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    return 0;
}
