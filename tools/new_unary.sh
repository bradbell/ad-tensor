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
#
if [ ! -e tools/$(basename $0) ]
then
    echo "$(basename $0) must be executed from its parent directory"
    exit 1
fi
# -----------------------------------------------------------------------------
# unary_function
if [ "$#" != 1 ]
then
    echo 'usage: tools/new_unary.sh unary_function'
    exit 1
fi
# List of files that need to change when adding a new unary operator:
#   include/ad_tensor/adten.hpp
#   include/ad_tensor/dev/op_enum.hpp
#   include/ad_tensor/dev/derive_op.hpp
#   src/CMakeLists.txt
#   src/dev/derive_op/op_enum2derive_op.cpp
#   src/dev/unary.cpp
#   src/dev/derive_op/<unary_function>_op.op
#   src/dev/to_string.cpp
#   src/examples/adten/unary.cpp
# -----------------------------------------------------------------------------
echo "$script_path: OK"
exit 0
