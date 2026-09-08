#! /usr/bin/env bash
set -e -u
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# script_path
script_dir="$( dirname -- "${BASH_SOURCE[0]}" )"
script_dir="$( cd -- "$script_dir" &> /dev/null && pwd )"
script_path="$script_dir/$(basename $0)"
# -----------------------------------------------------------------------------
cd build
ninja benchmarks
cd ..
echo "machine = $(uname -m)"
echo "kernel version = $(uname -r)"
echo "operating system - $(uname -o)"
#
configure='build/include/ad_tensor/configure.hpp'
compiler=$(grep CMAKE_CXX_COMPILER $configure | sed -e 's|.*COMPILER *||' )
$compiler --version | head -1
#
grep "CMAKE_BUILD_TYPE"  $configure
grep "AD_TENSOR_VERSION" $configure
grep "INCLUDE_PLUGIN"    $configure
#
#
benmchmarks='build/benchmarks/benchmarks'
$benmchmarks | sed -n \
    -e '/OK *\]/p' \
    -e '/^learn_ms =/p'
# -----------------------------------------------------------------------------
echo "$script_path: OK"
exit 0
