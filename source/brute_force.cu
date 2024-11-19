#include <brute_force.cuh>

void brute_force_gpu(const std::string &charset,
                     const unsigned char *target_hash,
                     const size_t max_length,
                     const std::vector<int> &hash_types_vec,
                     bool &found,
                     std::string &result)
{
    int charset_length = charset.length();
    unsigned long long total_candidates = pow(charset_length, max_length);
    const int *hash_types = &hash_types_vec[0];
    int hash_count = hash_types_vec.size();

    // CUDA-specific variables
    const int threads_per_block = 256;
    const unsigned long long chunk_size = 2L << 25;
    const unsigned long long num_chunks = (total_candidates + chunk_size - 1) / chunk_size;

    char *d_charset;
    unsigned char *d_target;
    bool *d_found;
    char *d_result;
    int *d_hash_types;

    cudaMalloc(&d_charset, charset_length * sizeof(char));
    cudaMalloc(&d_target, 32 * sizeof(unsigned char));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_result, max_length * sizeof(char));
    cudaMalloc(&d_hash_types, hash_count * sizeof(int));

    cudaMemcpy(d_charset, charset.c_str(), charset_length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_hash, 32 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &found, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_types, hash_types, hash_count * sizeof(int), cudaMemcpyHostToDevice);

    for (unsigned long long chunk = 0; chunk < num_chunks; ++chunk)
    {
        // std::cout << "Start chunk " << chunk << std::endl;
        unsigned long long start_idx = chunk * chunk_size;
        unsigned long long end_idx = std::min(start_idx + chunk_size, total_candidates);
        unsigned long long candidates_in_chunk = end_idx - start_idx;

        unsigned long long blocks_per_grid = (candidates_in_chunk + threads_per_block - 1) / threads_per_block;

        nested_hash_kernel<<<blocks_per_grid, threads_per_block>>>(d_charset,
                                                                   charset_length,
                                                                   max_length,
                                                                   start_idx,
                                                                   total_candidates,
                                                                   d_target,
                                                                   d_found,
                                                                   d_result,
                                                                   d_hash_types,
                                                                   hash_count);
        cudaDeviceSynchronize();

        cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
        if (found)
        {
            char result_char[16] = {0};
            cudaMemcpy(result_char, d_result, max_length * sizeof(char), cudaMemcpyDeviceToHost);
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

// CUDA Kernel for nested hashing
__global__ void nested_hash_kernel(char *charset,
                                   int charset_length,
                                   int max_len,
                                   unsigned long long start_idx,
                                   unsigned long long total_candidates,
                                   unsigned char *target,
                                   bool *found,
                                   char *result,
                                   int *hash_types,
                                   int hash_count)
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;

    if (idx >= total_candidates || *found)
    {
        return;
    }

    char candidate[16];
    unsigned long long current_idx = idx;
    for (int i = 0; i < max_len; ++i)
    {
        candidate[i] = charset[current_idx % charset_length];
        current_idx /= charset_length;
    }

    unsigned char intermediate[32]; // Buffer for intermediate hash resultssource
    unsigned char md5_hash[16];
    unsigned char sha1_hash[20];
    unsigned char sha256_hash[32];

    // Initialize with the candidate
    memcpy(intermediate, candidate, max_len);

    // Apply hash functions in sequence
    for (int i = 0; i < hash_count; ++i)
    {
        switch (hash_types[i])
        {
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

    if (equls_hash_gpu(intermediate, target, max_len))
    {
        *found = true;
        for (int i = 0; i < max_len; ++i)
        {
            result[i] = candidate[i];
        }
    }
}

__device__ bool equls_hash_gpu(const unsigned char *hash1, const unsigned char *hash2, int hash_len)
{
    for (int i = 0; i < hash_len; ++i)
    {
        if (hash1[i] != hash2[i])
        {
            return false;
        }
    }
    return true;
}

void brute_force_cpu(const std::string &charset,
                     const unsigned char *target_hash,
                     size_t max_length,
                     const std::vector<int> &hash_types,
                     bool &found,
                     std::string &result,
                     std::string current)
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    size_t hash_length;

    // Calculate the nested hash for the current combination
    calculate_nested_hash(current, hash_types, hash, hash_length);

    if (equls_hash(hash, target_hash, hash_length))
    {
        found = true;
    }

    if (current.size() == max_length)
    {
        return;
    }

    for (char c : charset)
    {
        if (found)
            return; // Stop further generation if a match is found
        brute_force_cpu(charset, target_hash, max_length, hash_types, found, result, current + c);
    }
}

void calculate_nested_hash(const std::string &input,
                           const std::vector<int> &hash_types,
                           unsigned char *hash,
                           size_t &hash_length)
{
    unsigned char intermediate_hash[SHA256_DIGEST_LENGTH]; // Largest hash buffer
    unsigned char current_input[SHA256_DIGEST_LENGTH];
    size_t current_input_len = input.size();

    // Copy the initial input into the current buffer
    memcpy(current_input, input.c_str(), input.size());

    for (int i = 0; i < hash_types.size(); ++i)
    {
        switch (hash_types[i])
        {
        case 0: // MD5
            MD5(current_input, current_input_len, intermediate_hash);
            current_input_len = MD5_DIGEST_LENGTH;
            break;
        case 1: // SHA1
            SHA1(current_input, current_input_len, intermediate_hash);
            current_input_len = SHA_DIGEST_LENGTH;
            break;
        case 2: // SHA256
            SHA256(current_input, current_input_len, intermediate_hash);
            current_input_len = SHA256_DIGEST_LENGTH;
            break;
        }
        memcpy(current_input, intermediate_hash, current_input_len);
    }

    // Copy the final hash to the output buffer
    memcpy(hash, intermediate_hash, current_input_len);
    hash_length = current_input_len;
}

bool equls_hash(const unsigned char *hash1, const unsigned char *hash2, int hash_len)
{
    for (int i = 0; i < hash_len; ++i)
    {
        if (hash1[i] != hash2[i])
        {
            return false;
        }
    }
    return true;
}

// Convert hex string to byte array
bool hex_to_bytes(const std::string &hex, unsigned char *bytes, int expected_length)
{
    if (hex.length() != expected_length * 2)
    {
        return false;
    }
    for (size_t i = 0; i < expected_length; ++i)
    {
        std::string byte_str = hex.substr(i * 2, 2);
        bytes[i] = static_cast<unsigned char>(std::stoi(byte_str, nullptr, 16));
    }
    return true;
}