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
if [ "$0" != "bin/check_build.sh" ]
then
   echo "bin/check_build.sh: must be executed from its parent directory"
   exit 1
fi
# -----------------------------------------------------------------------------
#
# run_cmake.sh
bin/run_cmake.sh
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
echo 'check_tests.sh: OK'
exit 0
