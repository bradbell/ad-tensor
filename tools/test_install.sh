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
if [ ! -e build/prefix/include/ad_tensor/ad_tensor.hpp ]
then
    echo 'test_install.sh: expected ad_tensor to install into build/prefix'
    exit 1
fi
#
# prefix
prefix=$(pwd)/build/prefix
#
# build/test_install
if [ -e build/test_install ]
then
    rm -r build/test_install
fi
mkdir build/test_install
cd build/test_install
#
# get_started.cpp
file="../../examples/get_started.cpp"
if ! grep 'TEST(examples, get_started)' $file
then
    echo 'test_install.sh:: cannot find TEST(examples, get_started) in'
    echo "$file"
    exit 1
fi
sed $file  > get_started.cpp \
    -e 's|TEST(examples, get_started)|int main(void)|' \
    -e 's|<gtest/gtest.h>|<cassert>|' \
    -e 's|EXPECT_TRUE|assert|' \
    -e 's|^}|    //\n    return 0;\n}|'
#
# CMakeLists.txt
cat << EOF > CMakeLists.txt
cmake_minimum_required(VERSION 3.12)
project( test_install )
#
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 20)
#
set(ad_tensor_DIR "$prefix/\${CMAKE_INSTALL_LIBDIR}/cmake/ad_tensor" )
find_package(ad_tensor REQUIRED)
#
set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} \${TORCH_CXX_FLAGS}")
#
include_directories( \${ad_tensor_INCLUDE_DIRS} \${TORCH_INCLUDE_DIRS} )
add_executable(get_started get_started.cpp)
target_link_libraries(get_started
    \${ad_tensor_LIBRARIES}
    \${TORCH_LIBRARIES}
)
EOF
#
# build
mkdir build
cmake -S . -B build -G Ninja
cmake --build build --target get_started
build/get_started
#
echo "$script_path: OK"
exit 0
