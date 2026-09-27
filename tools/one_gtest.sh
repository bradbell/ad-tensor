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
if [ $# != 2 ]
then
    echo 'tools/one_test.sh program gtest_name'
    echo 'where program is examples, tests, or benchmarks'
    echo 'and gtest_name is the test name without program_ at beginning'
    exit 1
fi
program="$1"
gtest_name="$2"
#
cd build
ninja $program
cd ..
echo build/$program/$program --gtest_filter="${program}_${gtest_name}"
build/$program/$program --gtest_filter="${program}_${gtest_name}"
# -----------------------------------------------------------------------------
echo "$script_path: OK"
exit 0
