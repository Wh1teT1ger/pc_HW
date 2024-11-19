#include <cuda_runtime.h>
#include <sha256.cuh>
#include <sha1.cuh>
#include <md5.cuh>
#include <string>

bool hex_to_bytes(const std::string& hex, unsigned char* bytes, int expected_length);

__global__ void nested_hash_kernel(char* charset, int charset_length, int max_len,
                                   unsigned long long start_idx, unsigned long long total_candidates,
                                   unsigned char* target, bool* found, char* result, int* hash_types, int hash_count);

void brute_force_gpu(std::string charset, unsigned char* target_hash, int max_len, std::vector<int> hash_sequence, std::string& result, bool& found);

void calculate_nested_hash(const std::string& input, const std::vector<std::string>& hash_sequence,
                           unsigned char* final_hash, size_t& final_hash_length);

bool equls_hash(const unsigned char* hash1, const unsigned char* hash2, int hash_len);

void brute_force_cpu(const std::string& charset, const size_t& max_length,
                           std::string current, const std::vector<std::string>& hash_sequence,
                           const unsigned char* target_hash , bool& found, std::string& found_word);

