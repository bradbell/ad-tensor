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
while [ $# -ge 1 ]
do
    case "$1" in
        #
        --release)
        release_flag='--release'
        ;;
        --no_plugin)
        plugin_flag='--no_plugin'
        ;;
        *)
cat << EOF
usage: tools/check_gtest.sh [--release] [--no_plugin]
EOF
        exit 1
    esac
    #
    shift
done
#
# run_cmake.sh
tools/run_cmake.sh $release_flag $plugin_flag
#
# n_job
if which nproc >& /dev/null
then
   n_job=$(nproc)
else
   n_job=$(sysctl -n hw.ncpu)
fi
if (( n_job > 1 ))
then
    n_job=$(( n_job - 1 ))
fi
#
# build
cd build
#
# build/examples/examples
echo_eval ninja -j $n_job examples
echo_eval ./examples/examples
#
# build/tests/tests
echo_eval ninja -j $n_job tests
echo_eval ./tests/tests
#
# build/benchmarks/benchmarks
echo_eval ninja -j $n_job benchmarks
echo_eval ./benchmarks/benchmarks
#
echo "$script_path: OK"
exit 0
