#! /usr/bin/env bash
set -e -u
# !! EDITS TO THIS FILE ARE LOST DURING UPDATES BY xrst.git/bin/dev_tools.sh !!
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
if [ "$0" != "bin/check_tests.sh" ]
then
   echo "bin/check_tests.sh: must be executed from its parent directory"
   exit 1
fi
# -----------------------------------------------------------------------------
# toch_dir
torch_dir=$(bin/torch_dir.py)
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
#
# cmake
echo_eval cmake -S .. -B . \
    -G Ninja \
    -D include_tests=true \
    -D Torch_DIR=$torch_dir
#
echo_eval ninja tests
echo_eval ./tests/tests
#
echo 'check_tests.sh: OK'
exit 0
