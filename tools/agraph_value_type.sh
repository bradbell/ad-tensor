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
# This script converts agraph.m_arg_value from vector<size_t> to
# vector<int64_t>. This may be a good idea because libtorch uses int64_t for
# many of its arguments (sometimes it even avoids the need to allocate an
# an array of ints). It may be a bad idea because these integers are always
# positive (and this restriction is lost when using int64_t.
# -----------------------------------------------------------------------------
if [ ! -e 'tools/agraph_value_type.sh' ]
then
    echo 'agraph_value_type.sh must be executed from its parent directory'
    exit 1
fi
git reset --hard
#
cat << EOF > temp.sed
s|vector<size_t> \\( *\\)m_arg_value;|vector<int64_t>\\1m_arg_value;|
EOF
#
file='include/ad_tensor/dev/agraph.hpp'
sed -i $file -f temp.sed
#
cat << EOF > temp.sed
s|m_arg_value  = ad_tensor::vector<size_t>( {0, 0} );|m_arg_value  = ad_tensor::vector<int64_t>( {0, 0} );|
EOF
file='tests/dev/base_op.cpp'
sed -i $file -f temp.sed
#
cat << EOF > temp.sed
s|size_t[*] begin|int64_t* begin|
s|size_t[*] end|int64_t* end|
s|vector<int64_t> *dim(begin, end)|c10::IntArrayRef dim(begin, end)|
s|vector<int64_t> *shape(begin, end)|c10::IntArrayRef shape(begin, end)|
EOF
list='
    src/dev/derive_op/flip_op.cpp
    src/dev/derive_op/sum_op.cpp
    src/dev/derive_op/view_op.cpp
    src/dev/derive_op/pad_op.cpp
'
for file in $list
do
    sed -i $file -f temp.sed
done
set +e
tools/check_version.sh
#
echo 'Run tools/check_all.sh to check that this conversion still works.'
#
echo "$script_path: OK"
