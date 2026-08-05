#! /usr/bin/env bash
set -e -u
# !! EDITS TO THIS FILE ARE LOST DURING UPDATES BY xrst.git/tools/dev_tools.sh !!
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
if [ ! -e "tools/run_cmake.sh" ]
then
   echo "tools/run_cmake.sh: must be executed from its parent directory"
   exit 1
fi
# -----------------------------------------------------------------------------
cmake_build_type='debug'
cmake_c_compiler='gcc'
cmake_cxx_compiler='g++'
while [ "$#" -ge 1 ]
do
    case $1 in

        '--release')
        cmake_build_type='release'
        ;;

        '--clang')
        cmake_c_compiler='clang'
        cmake_cxx_compiler='clang++'
        ;;

        *)
        echo 'usage: tools/run_cmake.sh [flag1 [flag2 .. ] ]'
        echo 'list of possible flags: --release'
        ;;

    esac
    shift
done
# -----------------------------------------------------------------------------
# toch_dir
torch_dir=$(tools/torch_dir.py)
#
# build
if [ ! -e build ]
then
    mkdir build
fi
cd build
if [ -e CMakeCache.txt ]
then
    rm -r CMakeCache.txt
fi
# temp.sed
# kineto library is not needed for cpu only version; see
# https://github.com/pytorch/pytorch/issues/110259
cat << EOF > temp.sed
/^ *CMake Warning at .*TorchConfig.cmake/ b one
/^ *CMake Warning at .*torch-config.cmake/ b one
b end
: one
N
/\\n *library kineto not found./! b end
:loop
N
/\\n *\$/! b loop
d
#
: end
EOF
#
# cxx_flags
cxx_flags='-std=c++23 -Wall -pedantic-errors -Wshadow -Wfloat-conversion -Wconversion'

#
# cmake
cat << EOF
cmake -S .. -B . \\
    -G Ninja \\
    -D include_tests=true \\
    -D CMAKE_BUILD_TYPE=$cmake_build_type \\
    -D Torch_DIR=$torch_dir \\
    -D CMAKE_CXX_FLAGS="'$cxx_flags'" \\
    -D CMAKE_C_COMPILER="$cmake_c_compiler" \\
    -D CMAKE_CXX_COMPILER="$cmake_cxx_compiler" \\
EOF
if ! cmake -S .. -B . \
    -G Ninja \
    -D include_tests=true \
    -D CMAKE_BUILD_TYPE=$cmake_build_type \
    -D Torch_DIR=$torch_dir \
    -D CMAKE_CXX_FLAGS="'$cxx_flags'" \
    -D CMAKE_C_COMPILER="$cmake_c_compiler" \
    -D CMAKE_CXX_COMPILER="$cmake_cxx_compiler" \
    2> temp.err
then
    sed -f temp.sed temp.err
    echo 'run_cmake.sh: errors in cmake output above'
    exit 1
fi
#
if sed -f temp.sed temp.err | grep 'CMake Warning' > /dev/null
then
    echo 'run_cmake.sh: warnings in cmake output above'
    exit 1
fi
#
echo "$script_path: OK"
exit 0
