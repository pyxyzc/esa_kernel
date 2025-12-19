cmake_minimum_required(VERSION 3.18)
project(esa LANGUAGES CXX CUDA)

# ---------- 1.  locate Python + pybind11 ------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
list(APPEND CMAKE_PREFIX_PATH ${Python_SITELIB}/pybind11)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)   # provides Torch::torch

if(DEFINED ENV{MY_INCLUDE_PATH})
    message(STATUS "Adding include path from MY_INCLUDE_PATH: $ENV{MY_INCLUDE_PATH}")
    # split colon‐separated paths into a CMake list
    string(REPLACE ":" ";" MY_INCLUDE_PATH_LIST "$ENV{MY_INCLUDE_PATH}")
    include_directories(${MY_INCLUDE_PATH_LIST})
endif()


# ---------- 2.  create the shared module ------------------
# pybind11_add_module already adds -fPIC, correct suffix, etc.
pybind11_add_module(esa
    esa_interface.cu      # contains PYBIND11_MODULE(esa, m){ ... }
    esa_kernels.cu
    cuda_sm_copy.cu
)

# ---------- 3.  compile flags -----------------------------
target_compile_features(esa PRIVATE cxx_std_17)
set_target_properties(esa PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON   # needed if kernels call each other
    POSITION_INDEPENDENT_CODE ON    # redundant but explicit
)

# ---------- 4.  include headers ----------------------------
target_include_directories(esa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ---------- 5.  link libraries -----------------------------
target_link_libraries(esa PRIVATE ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
target_compile_definitions(esa PRIVATE ${TORCH_COMPILE_DEFINITIONS})
target_include_directories(esa PRIVATE ${TORCH_INCLUDE_DIRS})
