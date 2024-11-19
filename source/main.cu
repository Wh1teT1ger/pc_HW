#include <iostream>
#include <vector>
#include <cmath>
#include <brute_force.cuh>
#include <argparse/argparse.hpp>
#include <chrono>

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("brute_force_nested_hashes");

    program.add_argument("-t", "target_hash")
        .help("The target hash to brute force")
        .required();

    program.add_argument("-h", "hash_sequence")
        .help("Sequence of hashes to apply (e.g., md5 sha1 sha256)").
        nargs(argparse::nargs_pattern::any).
        .required();


    program.add_argument("-c", "--charset")
        .help("Character set to use for generating combinations (default: abcdef)")
        .default_value(std::string("abcdefghijklmnopqrstuvwxyz"));

    program.add_argument("-m", "--max-length")
        .help("Maximum length of combinations (default: 6)")
        .scan<'i', int>()
        .default_value(6);

    program.add_argument("--gpu")
        .help("Choosing a gpu for brute force operation (default cpu)")
        .default_value(false)
        .implicit_value(true);



    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Parse arguments
    int max_len = program.get<int>("--max-length");
    std::string target_hash_hex = program.get<std::string>("-target_hash");
    std::vector<std::string> hash_sequence = program.get<std::vector<std::string>>("--hash-sequence");
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
    std::vector<int> hash_types; // Support up to 3 hash functions
    for (auto &hash : hash_sequence) {
        if (hash == "md5") hash_types.push_back(0);
        else if (hash == "sha1") hash_types.push_back(1);
        else if (hash == "sha256") hash_types.push_back(2);
        else {
            std::cerr << "Error: Unsupported hash type: " << hash_sequence[i] << std::endl;
            return 1;
        }
    }
    std::string result = "";
    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    bool found  = false;
    std::string reuslt = "";
    if (program["--gpu"] == true) {
        brute_force_gpu(charset_str, target_hash, max_len, hash_types, found, result);
    } else {
        brute_force_cpu(charset_str, target_hash, max_len, hash_types, found, reuslt, "");
    }
    
    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (found) {
        std::cout << "Match found: " << result << std::endl;
    } else {
        std::cout << "No match found." << std::endl;
    }

    

    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    return 0;
}
