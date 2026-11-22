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
if [ $# != 1 ] && [ $# != 2 ]
then
    echo 'tools/one_test.sh gtest_name [--no_plugin]'
    exit 1
fi
gtest_name="$1"
build_dir='build/yes_plugin'
if [ $# == 2 ]
then
    if [ "$2" != '--no_plugin' ]
    then
        echo 'tools/one_test.sh gtest_name [--no_plugin]'
        exit 1
    fi
    build_dir='build/no_plugin'
fi
#
# progran
program=$(echo $gtest_name | sed -e 's|[_.].*||')
#
# n_job
if ! n_job=$(n_compile_job.sh)
then
    echo $n_job
    exit 1
fi
#
echo "ninja -C $build_dir -j $n_job $program"
ninja -C $build_dir -j $n_job $program
#
echo $build_dir/$program/$program --gtest_filter=$gtest_name
$build_dir/$program/$program --gtest_filter=$gtest_name
# -----------------------------------------------------------------------------
echo "$script_path: OK"
exit 0
