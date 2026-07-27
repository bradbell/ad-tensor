#! /usr/bin/env bash
set -e -u
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
#
# script_path
script_dir="$( dirname -- "${BASH_SOURCE[0]}" )"
script_dir="$( cd -- "$script_dir" &> /dev/null && pwd )"
script_path="$script_dir/$(basename $0)"
# -----------------------------------------------------------------------------
if [ ! -e 'tools/check_compare.sh' ]
then
    echo "tools/check_compare.sh: must be executed from its parent directory"
    exit 1
fi
#
# sed
source tools/grep_and_sed.sh
#
declare -A name2op
name2op['lt']='<'
name2op['le']='<='
name2op['eq']='=='
name2op['ne']='!='
name2op['ge']='>='
name2op['gt']='>'
#
gt_file='src/dev/derive_op/gt_op.cpp'
#
for name in lt le eq ne ge
do
    out_file=$(echo $gt_file | sed -e "s|gt_op|${name}_op|")
    sed $gt_file > temp.$$ \
    -e "s|gt_op|${name}_op|g" \
    -e "s|lhs_tensor *> *rhs_tensor|lhs_tensor ${name2op[$name]} rhs_tensor|"
    if ! diff $out_file temp.$$
    then
        echo "check_compare.sh: The file $out_file"
        echo "Does not agree with the file $gt_file"
        rm temp.$$
        exit 1
    fi
done
rm temp.$$
#
echo "$script_path: OK"
