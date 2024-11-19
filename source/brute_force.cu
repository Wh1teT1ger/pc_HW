#include <brute_force.cuh>

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
    for (int i = 0; i < max_len; ++i) {
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
void brute_force_gpu(std::string charset, unsigned char* target_hash, int max_len, std::vector<int> hash_sequence, std::string& result, bool& found) {
    int charset_length = charset.length();
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
    

    cudaMalloc(&d_charset, charset_length * sizeof(char));
    cudaMalloc(&d_target, 32 * sizeof(unsigned char));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_result, max_len * sizeof(char));
    cudaMalloc(&d_hash_types, hash_count * sizeof(int));

    cudaMemcpy(d_charset, charset_str.c_str(), charset_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_hash, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &found, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_types, hash_types, hash_count * sizeof(int), cudaMemcpyHostToDevice);

    for (unsigned long long chunk = 0; chunk < num_chunks; ++chunk) {
        //std::cout << "Start chunk " << chunk << std::endl;
        unsigned long long start_idx = chunk * chunk_size;
        unsigned long long end_idx = std::min(start_idx + chunk_size, total_candidates);
        unsigned long long candidates_in_chunk = end_idx - start_idx;

        unsigned long long blocks_per_grid = (candidates_in_chunk + threads_per_block - 1) / threads_per_block;

        nested_hash_kernel<<<blocks_per_grid, threads_per_block>>>(d_charset, charset_length, max_len,
                                                                   start_idx, total_candidates, d_target, d_found, d_result, d_hash_types, hash_count);
        cudaDeviceSynchronize();

        cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        if (found) {
            char result_char[16] = {0};
            cudaMemcpy(result, d_result, max_len * sizeof(char), cudaMemcpyDeviceToHost);
            result = result_char;
            break;
        }
    }

    cudaFree(d_charset);
    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_result);
    cudaFree(d_hash_types);
}