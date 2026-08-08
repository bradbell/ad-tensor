#! /usr/bin/env bash
set -e -u
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
# SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
# SPDX-FileContributor: 2026 Bradley M. Bell
# -----------------------------------------------------------------------------
# List of files that need to change when adding a new unary operator:
#
#   src/dev/derive_op/<fun_name>_op.cpp
#   src/examples/adten/unary.cpp
#
#   include/ad_tensor/adten.hpp
#   include/ad_tensor/dev/op_enum.hpp
#   include/ad_tensor/dev/derive_op.hpp
#   src/adten/unary.cpp
#   src/CMakeLists.txt
#   src/dev/derive_op/op_enum2derive_op.cpp
#   src/dev/to_string.cpp
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
# fun_name, chars_minus_3
if [ "$#" != 2 ] 
then
    echo 'usage:       tools/new_unary.sh fun_name description'
    echo 'fun_name:    is the funciton name; e.g., exp'
    echo 'description: is the function description; e.g., exponentiation'
    exit 1
fi
fun_name="$1"
description="$2"
chars_minus_3=$(( ${#1} - 3 ))
# -----------------------------------------------------------------------------
file_in=src/dev/derive_op/exp_op.cpp
file_out=src/dev/derive_op/${fun_name}_op.cpp
cat << EOF > temp.sed
s|exp_op|${fun_name}_op|
s|exp()|$fun_name()|
EOF
echo "$file_out"
sed -f temp.sed $file_in > $file_out
git add $file_out
# -----------------------------------------------------------------------------
file='examples/adten/unary.cpp'
cat << EOF > temp.sed
/TEST(.*, unary_exp)/! b end
: loop
N
/\\n}/! b loop 
h
s|^|//\\n// unary_$fun_name\\n|
s|exp|$fun_name|g
H
x
: end
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='include/ad_tensor/adten.hpp'
cat << EOF > temp.sed
s|^\\( *\\)AD_TENSOR_UNARY_OP(exp);|&\\
\\1AD_TENSOR_UNARY_OP($fun_name);|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='include/ad_tensor/dev/op_enum.hpp'
cat << EOF > temp.sed
s|^\\( *\\)exp, \\{$chars_minus_3\\}\\( *\\)//.*|&\\
\\1$fun_name,\\2// $description|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='include/ad_tensor/dev/derive_op.hpp'
cat << EOF > temp.sed
s|^\\( *\\)AD_TENSOR_DERIVE_OP(exp_op)|&\\
\\1AD_TENSOR_DERIVE_OP(${fun_name}_op)|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/adten/unary.cpp'
cat << EOF > temp.sed
/[/][/] exp/! b end
N
N
N
N
s|^\\( *\\).*|&\\
\\1// $fun_name\\
\\1case dev::op_enum_t::$fun_name:\\
\\1res_at_ten = operand.at_ten().$fun_name();\\
\\1break;\\
\\1//|
: end
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/CMakeLists.txt'
cat << EOF > temp.sed
s|^\\( *\\)dev/derive_op/exp_op.cpp|&\\
\\1dev/derive_op/${fun_name}_op.cpp|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/dev/derive_op/op_enum2derive_op.cpp'
cat << EOF > temp.sed
s|^\\( *static const \\)exp_op_t<TensorType> \\{$chars_minus_3\\}\\( *\\).*|&\\
\\1${fun_name}_op_t<TensorType>\\2${fun_name}_op;|
s|^\\( *case op_enum_t::\\)exp: \\{$chars_minus_3\\}\\( *\\).*|&\\
\\1${fun_name}:\\2return ${fun_name}_op;|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
file='src/dev/to_string.cpp'
cat << EOF > temp.sed
s|^\\( *\\)case op_enum_t::exp: \\{$chars_minus_3\\}\\( *\\).*|&\\
\\1case op_enum_t::${fun_name}:\\2return "${fun_name}";|
EOF
git checkout --quiet $file
echo "$file"
sed -i $file -f temp.sed
# -----------------------------------------------------------------------------
tools/run_cmake.sh
echo
echo 'The following files require editing by hand:'
echo src/dev/derive_op/${fun_name}_op.cpp
echo examples/adten/unary.cpp
# -----------------------------------------------------------------------------

echo "$script_path: OK"
exit 0
