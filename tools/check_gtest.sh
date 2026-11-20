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
#
# release_flag, plugin_flag
release_flag=''
plugin_flag=''
clang_flag=''
build_dir='build/yes_plugin'
while [ $# -ge 1 ]
do
    case "$1" in
        #
        --release)
        release_flag='--release'
        ;;
        --no_plugin)
        plugin_flag='--no_plugin'
        build_dir='build/no_plugin'
        ;;
        --clang)
        clang_flag='--clang'
        ;;
        *)
cat << EOF
usage: tools/check_gtest.sh [--release] [--no_plugin] [--clang]
EOF
        exit 1
    esac
    #
    shift
done
#
# run_cmake.sh
tools/run_cmake.sh $release_flag $plugin_flag $clang_flag
#
# n_job
if ! n_job=$(tools/n_compile_job.sh)
then
    echo $n_job
    exit 1
fi
#
# $build_dir
cd $build_dir
#
# $build_dir/examples/examples
echo_eval ninja -j $n_job examples
echo_eval ./examples/examples
#
# $build_dir/tests/tests
echo_eval ninja -j $n_job tests
echo_eval ./tests/tests
#
# $build_dir/benchmarks/benchmarks
echo_eval ninja -j $n_job benchmarks
echo_eval ./benchmarks/benchmarks
#
echo "$script_path: OK"
exit 0
