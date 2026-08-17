#! /usr/bin/env bash
include/ad_tensor/dev/op_enum.hpp
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
if [ ! -e tools/$(basename $0) ]
then
    echo "$(basename $0) must be executed from its parent directory"
    exit 1
fi
# -----------------------------------------------------------------------------
# name_new, chars_minus_3
if [ "$#" != 2 ]
then
    echo 'usage:       tools/new_unary.sh old_name new_name'
    echo 'name_old:    is the name of an already implemented function'
    echo 'name_new:    is the name of function we are implementing'
    exit 1
fi
name_old="$1"
name_new="$2"
NAME_OLD=$(echo $name_old | tr [a-z] [A-Z])
NAME_NEW=$(echo $name_new | tr [a-z] [A-Z])
chars_minus_3=$(( ${#2} - 3 ))
# -----------------------------------------------------------------------------
file_old=src/adten/${name_old}.cpp
file_new=src/adten/${name_new}.cpp
cat << EOF > temp.sed
s|${name_old}|${name_new}|g
s|${NAME_OLD}|${NAME_NEW}|g
EOF
git checkout --quiet $file_old
echo "$file_new"
sed -f temp.sed $file_old > $file_new
git add $file_new
# -----------------------------------------------------------------------------
file_old=src/dev/derive_op/${name_old}_op.cpp
file_new=src/dev/derive_op/${name_new}_op.cpp
cat << EOF > temp.sed
s|${name_old}_op|${name_new}_op|
s|$name_old()|$name_new()|
EOF
git checkout --quiet $file_old
echo "$file_new"
sed -f temp.sed $file_old > $file_new
git add $file_new
# -----------------------------------------------------------------------------
file_old=examples/adten/$name_old.cpp
file_new=examples/adten/$name_new.cpp
cat << EOF > temp.sed
s|$name_old|$name_new|g
EOF
git checkout --quiet $file_old
echo "$file_new"
sed -f temp.sed $file_old > $file_new
git add $file_new
# -----------------------------------------------------------------------------
file='examples/CMakeLists.txt'
cat << EOF > temp.sed
s|^\\( *\\)adten/$name_old.cpp|&\\
\\1aten/${name_new}.cpp|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='include/ad_tensor/adten.hpp'
cat << EOF > temp.sed
s|^}; }\$|    //\\
    // $name_new\\
    adten_t $name_new(\\
    );\\
&|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='include/ad_tensor/dev/op_enum.hpp'
cat << EOF > temp.sed
s|BEGIN_OTHER BEGIN.*|&\\
    $name_new,|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='include/ad_tensor/dev/derive_op.hpp'
cat << EOF > temp.sed
s|^\\( *\\)AD_TENSOR_DERIVE_OP(${name_old}_op)|&\\
\\1AD_TENSOR_DERIVE_OP(${name_new}_op)|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/CMakeLists.txt'
cat << EOF > temp.sed
s|^\\( *\\)adten/${name_old}.cpp|&\\
\\1adten/${name_new}.cpp|
s|^\\( *\\)dev/derive_op/${name_old}_op.cpp|&\\
\\1dev/derive_op/${name_new}_op.cpp|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/dev/derive_op/op_enum2derive_op.cpp'
cat << EOF > temp.sed
s|^\\( *static const \\)exp_op_t<TensorType> \\{$chars_minus_3\\}\\( *\\).*|&\\
\\1${name_new}_op_t<TensorType>\\2${name_new}_op;|
s|^\\( *case op_enum_t::\\)exp: \\{$chars_minus_3\\}\\( *\\).*|&\\
\\1${name_new}:\\2return ${name_new}_op;|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/dev/to_string.cpp'
cat << EOF > temp.sed
s|^\\( *\\)case op_enum_t::exp: \\{$chars_minus_3\\}\\( *\\).*|&\\
\\1case op_enum_t::${name_new}:\\2return "${name_new}";|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
set +e
tools/check_sort.sh
tools/run_cmake.sh
set -e
cat << EOF

Changes to these files should not need editing:
src/dev/derive_op/op_enum2derive_op.cpp
include/ad_tensor/dev/derive_op.hpp
examples/CMakeLists.txt
src/CMakeLists.txt
src/dev/to_string.cpp

Edit the files below in the following order:
src/adten/${name_new}.cpp
include/ad_tensor/adten.hpp
include/ad_tensor/dev/op_enum.hpp

src/dev/derive_op/${name_new}_op.cpp
src/examples/adten/${name_new}.cpp

EOF
echo "$script_path: OK"
exit 0
