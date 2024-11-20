#include <iostream>
#include <vector>
#include <argparse/argparse.hpp>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <cstring>

void to_hex(const unsigned char* hash, size_t length, char* hex_str) {
    const char hex_chars[] = "0123456789abcdef";
    for (size_t i = 0; i < length; ++i) {
        hex_str[i * 2] = hex_chars[(hash[i] >> 4) & 0xF];  // High nibble
        hex_str[i * 2 + 1] = hex_chars[hash[i] & 0xF];     // Low nibble
    }
    hex_str[length * 2] = '\0';
}



int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("hash");

    program.add_argument("-t", "--target")
        .help("The target to hash")
        .required();

    program.add_argument("-s", "--hash_sequence")
        .help("Sequence of hashes to apply (e.g., md5 sha1 sha256)").
        nargs(argparse::nargs_pattern::any)
        .required();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Parse arguments
    std::string target = program.get<std::string>("--target");
    std::vector<std::string> hash_sequence = program.get<std::vector<std::string>>("--hash_sequence");
    unsigned char buffer[SHA256_DIGEST_LENGTH];
    memcpy(buffer, target.c_str(), target.size());
    size_t current_len = target.size();

    // Perform nested hashing based on the sequence
    for (int i = 0; i < hash_sequence.size(); ++i) {
        if (hash_sequence[i] == "md5") {
            MD5(buffer, current_len, buffer);
            current_len = MD5_DIGEST_LENGTH;
        } else if (hash_sequence[i] == "sha1") {
            SHA1(buffer, current_len, buffer);
            current_len = SHA_DIGEST_LENGTH;
        } else if (hash_sequence[i] == "sha256") {
            SHA256(buffer, current_len, buffer);
            current_len = SHA256_DIGEST_LENGTH;
        }
    }

    char result[SHA256_DIGEST_LENGTH * 2 + 1];

    // Convert the final hash to hexadecimal and store in result
    to_hex(buffer, current_len, result);

    std::cout << result << std::endl;

    return 0;
}
