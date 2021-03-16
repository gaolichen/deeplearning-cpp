# CMakeLists files in this project can

cmake_minimum_required (VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#SET(MYDEV D:/dev)
SET(MYDEV /home/gaolichen/mydev)
SET(BOOST_ROOT ${MYDEV}/boost_1_70_0)

project (dpl)

# DO NOT CALL find_* before defining project,
# PROJECT sets some important variables about the plarform.
find_package(Boost 1.70)
find_package(Python2 COMPONENTS Development NumPy)
find_package(OpenMP)

SET(BLAS_LIBRARIES /opt/OpenBLAS/lib/libopenblas.so)
#set(BLA_VENDER OpenBLAS)
#find_package(BLAS REQUIRED HINTS $ENV{BLAS_HOME})
#if(BLAS_FOUND)
#    message(STATUS "OpenBLAS library=${BLAS_LIBRARIES}") 
#endif(BLAS_FOUND)
message(STATUS "OpenBLAS library=${BLAS_LIBRARIES}")

SET (CMAKE_BUILD_TYPE Release)
# to use gdb enable the following two flags.
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_subdirectory (src)

