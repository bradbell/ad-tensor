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
# name_new, name_other, chars_minus_2
if [ "$#" != 2 ]
then
    echo 'usage:       tools/new_unary.sh old_name new_name'
    echo 'name_new:    is the name of operator we are implementing'
    echo 'name_other:  is the name of a similar already implemented operator'
    exit 1
fi
name_new="$1"
name_other="$2"
NAME_OTHER=$(echo $name_other | tr [a-z] [A-Z])
NAME_NEW=$(echo $name_new | tr [a-z] [A-Z])
chars_minus_2=$(( ${#name_new} - 2 ))
# -----------------------------------------------------------------------------
# src/adten/$name_new.cpp
file_other=src/adten/${name_other}.cpp
file_new=src/adten/${name_new}.cpp
cat << EOF > temp.sed
s|${name_other}|${name_new}|g
s|${NAME_OTHER}|${NAME_NEW}|g
EOF
git checkout --quiet $file_other
sed -f temp.sed $file_other > $file_new
git add $file_new
echo "$file_new"
# -----------------------------------------------------------------------------
# src/dev/derive_op/${name_new}_op.cpp
file_other=src/dev/derive_op/${name_other}_op.cpp
file_new=src/dev/derive_op/${name_new}_op.cpp
cat << EOF > temp.sed
s|${name_other}_op|${name_new}_op|
s|$name_other()|$name_new()|
EOF
git checkout --quiet $file_other
sed -f temp.sed $file_other > $file_new
git add $file_new
echo "$file_new"
# -----------------------------------------------------------------------------
# examples/adten/$name_new.cpp
file_other=examples/adten/$name_other.cpp
file_new=examples/adten/$name_new.cpp
cat << EOF > temp.sed
s|$name_other|$name_new|g
EOF
git checkout --quiet $file_other
sed -f temp.sed $file_other > $file_new
git add $file_new
echo "$file_new"
# -----------------------------------------------------------------------------
# tests/adten/$name_new.cpp
file_other=tests/adten/$name_other.cpp
file_new=tests/adten/$name_new.cpp
cat << EOF > temp.sed
s|$name_other|$name_new|g
EOF
if [ -e $file_other ]
then
    git checkout --quiet $file_other
    sed -f temp.sed $file_other > $file_new
    git add $file_new
    echo "$file_new"
fi
# -----------------------------------------------------------------------------
# examples/CMakeLists.txt
file='examples/CMakeLists.txt'
cat << EOF > temp.sed
s|^\\( *\\)adten/$name_other.cpp|&\\
\\1adten/${name_new}.cpp|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
# tests/CMakeLists.txt
file='tests/CMakeLists.txt'
cat << EOF > temp.sed
s|^\\( *\\)adten/$name_other.cpp|&\\
\\1adten/${name_new}.cpp|
EOF
if [ -e tests/adten/$name_new.cpp ]
then
    git checkout --quiet $file
    sed -i $file -f temp.sed
    echo "$file"
fi
# -----------------------------------------------------------------------------
# include/ad_tensor/adten.hpp
file='include/ad_tensor/adten.hpp'
cat << EOF > temp.sed
s|^}; }\$|    //\\
    // $name_new\\
    adten_t $name_new(\\
    );\\
&|
s|\\(^ *\\)src/adten/$name_old.cpp|&\\
\\1src/adten/$name_new.cpp|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
# include/ad_tensor/dev/op_enum.hpp
file='include/ad_tensor/dev/op_enum.hpp'
cat << EOF > temp.sed
s|BEGIN_OTHER BEGIN.*|&\\
    $name_new,|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
# include/ad_tensor/dev/derive_op.hpp
file='include/ad_tensor/dev/derive_op.hpp'
cat << EOF > temp.sed
s|^\\( *\\)AD_TENSOR_DERIVE_OP(${name_other}_op)|&\\
\\1AD_TENSOR_DERIVE_OP(${name_new}_op)|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
# src/CMakeLists.txt
file='src/CMakeLists.txt'
cat << EOF > temp.sed
s|^\\( *\\)adten/${name_other}.cpp|&\\
\\1adten/${name_new}.cpp|
s|^\\( *\\)dev/derive_op/${name_other}_op.cpp|&\\
\\1dev/derive_op/${name_new}_op.cpp|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
# src/dev/enum2derive.cpp
file='src/dev/enum2derive.cpp'
cat << EOF > temp.sed
s|^\\( *static const \\)lt_op_t<TensorType> \\{$chars_minus_2\\}\\( *\\).*|&\\
\\1${name_new}_op_t<TensorType>\\2${name_new}_op;|
s|^\\( *case op_enum_t::\\)lt: \\{$chars_minus_2\\}\\( *\\).*|&\\
\\1${name_new}:\\2return ${name_new}_op;|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
# src/dev/to_string.cpp
file='src/dev/to_string.cpp'
cat << EOF > temp.sed
s|^\\( *\\)case op_enum_t::lt: \\{$chars_minus_2\\}\\( *\\).*|&\\
\\1case op_enum_t::${name_new}:\\2return "${name_new}";|
EOF
git checkout --quiet $file
sed -i $file -f temp.sed
echo "$file"
# -----------------------------------------------------------------------------
set +e
tools/check_sort.sh
tools/run_cmake.sh
set -e
cat << EOF
Changes to these files should not need editing:
src/dev/enum2derive.cpp
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
