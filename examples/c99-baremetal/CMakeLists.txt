cmake_minimum_required(VERSION 3.10)
project(c99_baremetal_inference C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(c99_baremetal_inference src/main.c)
