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
# -----------------------------------------------------------------------------
if [ "$0" != "tools/run_cmake.sh" ]
then
   echo "tools/run_cmake.sh: must be executed from its parent directory"
   exit 1
fi
# -----------------------------------------------------------------------------
cmake_build_type='debug'
while [ "$#" -ge 1 ]
do
    case $1 in

        '--release')
        cmake_build_type='release'
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
cxx_flags='-Wall -pedantic-errors -Wshadow -Wfloat-conversion -Wconversion'

#
# cmake
echo_eval cmake -S .. -B . \
    -G Ninja \
    -D include_tests=true \
    -D CMAKE_BUILD_TYPE=$cmake_build_type \
    -D Torch_DIR=$torch_dir \
    -D CMAKE_CXX_FLAGS="'$cxx_flags'" \
    |& sed -f temp.sed  > temp.out
#
cat temp.out
if grep 'CMake Warning' temp.out > /dev/null
then
    echo 'run_cmake.sh: Warnings in cmake output above'
    exit 1
fi
#
echo 'run_cmake.sh: OK'
exit 0
