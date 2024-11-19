#include <cuda_runtime.h>
#include <sha256.cuh>
#include <sha1.cuh>
#include <md5.cuh>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <string>

void brute_force_gpu(const std::string &charset,
                     const unsigned char *target_hash,
                     const size_t max_length,
                     const std::vector<int> &hash_types,
                     bool &found,
                     std::string &result);

__global__ void nested_hash_kernel(char *charset,
                                   int charset_length,
                                   int max_len,
                                   unsigned long long start_idx,
                                   unsigned long long total_candidates,
                                   unsigned char *target,
                                   bool *found,
                                   char *result,
                                   int *hash_types,
                                   int hash_count);

void brute_force_cpu(const std::string &charset,
                     const unsigned char *target_hash,
                     size_t max_length,
                     const std::vector<int> &hash_types,
                     bool &found,
                     std::string &result,
                     std::string current);

void calculate_nested_hash(const std::string &input,
                           const std::vector<std::string> &hash_sequence,
                           unsigned char *hash,
                           size_t &hash_length);

bool equls_hash(const unsigned char *hash1, const unsigned char *hash2, int hash_len);

bool hex_to_bytes(const std::string &hex, unsigned char *bytes, int expected_length);
