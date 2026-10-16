#! /usr/bin/env bash
set -e -u
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# echo_eval
echo_eval() {
   echo $*
   eval $*
}
#
# script_path
script_dir="$( dirname -- "${BASH_SOURCE[0]}" )"
script_dir="$( cd -- "$script_dir" &> /dev/null && pwd )"
script_path="$script_dir/$(basename $0)"
# -----------------------------------------------------------------------------
if [ ! -e "tools/test_install.sh" ]
then
   echo "tools/test_install.sh: must be executed from its parent directory"
   exit 1
fi
# -----------------------------------------------------------------------------
if [ ! -e build/CMakeCache.txt ]
then
    echo 'test_install.sh: must first create build directory and run cmake'
    exit 1
fi
#
# install
if [ -d build/prefix/include/ad_tensor ]
then
    rm -r build/prefix/include/ad_tensor
fi
cmake --build build --target install
#
# repository
repository=$(pwd)
#
# prefix
prefix=$repository/build/prefix
#
if [ ! -e $prefix/include/ad_tensor/ad_tensor.hpp ]
then
    echo 'test_install.sh: expected ad_tensor to install with preifx'
    echo "$prefix"
    exit 1
fi
#
# ad_tensor_DIR
ad_tensor_DIR=''
for libdir in lib lib64
do
    if [ -e "$prefix/$libdir/cmake/ad_tensor/ad_tensorConfig.cmake" ]
    then
        ad_tensor_DIR="$prefix/$libdir/cmake/ad_tensor"
    fi
done
if [ "$ad_tensor_DIR" == '' ]
then
    echo 'Cannot find ad_tensorConfig.cmake below prefix'
    echo $prefix
    exit 1
fi
#
# build/test_install
if [ -e build/test_install ]
then
    rm -r build/test_install
fi
mkdir build/test_install
cd build/test_install
#
# CMakeLists.txt
cat << 'EOF' > CMakeLists.txt
# BEGIN_CMAKELISTS_TXT_EXAMPLE
# This will compile and link the file get_started.cpp
cmake_minimum_required(VERSION 3.12)
project( ad_tensor_example )
set(CMAKE_CXX_STANDARD 20)
#
find_package(ad_tensor REQUIRED)
#
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ad_tensor_CXX_FLAGS}")
include_directories( ${ad_tensor_INCLUDE_DIRS} )
add_executable(example get_started.cpp)
target_link_libraries(example ${ad_tensor_LIBRARIES} )
# END_CMAKELISTS_TXT_EXAMPLE
EOF
#
# BEGIN_BUILD_AND_RUN_EXAMPLE
# The directory $repository is the ad_tensor git repository on this machine.
# The directory $ad_tensor_DIR contains the file ad_tensorConfig.cmake.
mkdir build
cp $repository/examples/get_started.cpp get_started.cpp
cmake -S . -B build -G Ninja \
    -D "ad_tensor_DIR=$ad_tensor_DIR" \
    -D "CMAKE_BUILD_TYPE=debug"
cmake --build build --target example
build/example
# END_BUILD_AND_RUN_EXAMPLE
#
echo "$script_path: OK"
exit 0
