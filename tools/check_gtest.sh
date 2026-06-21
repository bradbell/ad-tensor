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
if [ ! -e 'tools/check_gtest.sh' ]
then
   echo "tools/check_gtest.sh: must be executed from its parent directory"
   exit 1
fi
# -----------------------------------------------------------------------------
#
# run_cmake.sh
tools/run_cmake.sh
#
# build
cd build
#
# build/examples/examples
echo_eval ninja examples
echo_eval ./examples/examples
#
# build/tests/tests
echo_eval ninja tests
echo_eval ./tests/tests
#
echo "$script_path: OK"
exit 0
