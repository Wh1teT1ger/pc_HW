cmake_minimum_required(VERSION 3.10)
project(nested_hash LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define directories
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")

# fetch latest argparse
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# Add the sha1 library
add_library(sha1 STATIC ${SOURCE_DIR}/sha1.cu)
target_include_directories(sha1 PUBLIC ${INCLUDE_DIR})
set_target_properties(sha1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add the sha256 library
add_library(sha256 STATIC ${SOURCE_DIR}/sha256.cu)
target_include_directories(sha256 PUBLIC ${INCLUDE_DIR})
set_target_properties(sha256 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add the md5 library
add_library(md5 STATIC ${SOURCE_DIR}/md5.cu)
target_include_directories(md5 PUBLIC ${INCLUDE_DIR})
set_target_properties(md5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add the sha1 library
add_library(brute_force STATIC ${SOURCE_DIR}/brute_force.cu)
target_include_directories(brute_force PUBLIC ${INCLUDE_DIR})
set_target_properties(brute_force PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Add the main CUDA executable
add_executable(nested_hash ${SOURCE_DIR}/main.cu)
set_target_properties(nested_hash PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Include directories for the main executable
target_link_libraries(nested_hash PRIVATE sha1 sha256 md5 argparse brute_force)

# Include directories for the main executable
target_include_directories(nested_hash PRIVATE ${INCLUDE_DIR})