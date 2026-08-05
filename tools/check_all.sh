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
if [ ! -e 'tools/check_all.sh' ]
then
    echo "tools/check_all.sh: must be executed from its parent directory"
    exit 1
fi
#
# sed
source tools/grep_and_sed.sh
#
# skip_check_copy, external_links
external_links='no'
skip_check_copy='no'
release='no'
while [ $# -ge 1 ]
do
    case "$1" in
        #
        --skip_check_copy)
        skip_check_copy='no'
        ;;
        --external_links)
        external_links='yes'
        ;;
        --release)
        release='yes'
        ;;
        *)
cat << EOF
usage: tools/check_all.sh [-skip_check_copy] [--external_links] [--release]
EOF
        exit 1
    esac
    #
    shift
done
# -----------------------------------------------------------------------------
#
# typos
if which typos > /dev/null
then
     echo_eval typos
fi
#
# run_xrsth.sh
if [ "$external_links" == 'no' ]
then
    echo_eval tools/run_xrst.sh
else
    echo_eval tools/run_xrst.sh --external_links
fi
#
# check_copy.sh
if [ "$skip_check_copy" == 'no' ]
then
    tools/check_copy.sh
fi
#
# check_list
check_list=$(ls tools/check_* | $sed \
    -e '/^tools[/]check_all.sh/d' \
    -e '/^tools[/]check_copy.sh/d' \
    -e '/^tools[/]check_gtest.sh/d' \
)
for check in $check_list
do
    echo_eval $check
done
#
# check_gtest.sh
# Do last because other tests may change things that require this to re-run.
if [ "$release" == 'yes' ]
then
    echo_eval check_gtest.sh --release
else
    echo_eval check_gtest.sh
fi
#
echo "$script_path: OK"
exit 0
