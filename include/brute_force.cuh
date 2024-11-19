#include <cuda_runtime.h>
#include <sha256>
#include <sha1>
#include <md5>
#include <string>

bool hex_to_bytes(const std::string& hex, unsigned char* bytes, int expected_length);

__global__ void nested_hash_kernel(char* charset, int charset_length, int max_len,
                                   unsigned long long start_idx, unsigned long long total_candidates,
                                   unsigned char* target, bool* found, char* result, int* hash_types, int hash_count);

void brute_force_gpu(std::string charset, unsigned char* target_hash, int max_len, std::vector<int> hash_sequence, std::string& result, bool& found);

