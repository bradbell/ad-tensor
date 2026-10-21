// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
// SPDX-FileCopyrightText: Bradley M. Bell <bradbell@seanet.com>
// SPDX-FileContributor: 2026 Bradley M. Bell
// ----------------------------------------------------------------------------
/*
{xrst_begin src_gen_index_array dev}
{xrst_spell
    op
    newline
}

Generate Source Code Index Array Operators
##########################################

Syntax
******
{xrst_code cpp}
    src = src_gen_index_array(op_index, agraph, variable_agraph, tensor_src);
{xrst_code}

Prototype
*********
{xrst_literal ,
    BEGIN_SRC_GEN_INDEX_ARRAY, END_SRC_GEN_INDEX_ARRAY
}

op_index
********
is the index of this operator in the operation sequence.
It is also the index of the result for this operator.
It must be greater than zero, because the first result is an
independent parameter tensor.

op_adtype
*********
is the type of the operator sequence; i.e., parameter or variable.

agraph
******
Is the :ref:`agraph-name` for the operation sequence that we are using.


tensor_src
**********
The function call
{xrst_code cpp}
    src = tensor_src(index, adtype)
{xrst_code}
returns a source code representation of a
constant, parameter of variable (depending on adtype) with the specified index.
The return, src, can be used to set or get the corresponding value.
There are no newline characters in src; hence it is one line of source code.

src
===
Is the source code corresponding to the index array operator at index op_index.

{xrst_end src_gen_index_array}
*/
#include <cassert>
#include <format>
#include <string>
#include <functional>
#include <iostream>
#include <ad_tensor/adtype.hpp>
#include <ad_tensor/dev/agraph.hpp>
#include <ad_tensor/dev/op_enum.hpp>
#include <ad_tensor/dev/to_string.hpp>
//
namespace ad_tensor { namespace dev { // ad_tensor::dev
//
// BEGIN_SRC_GEN_INDEX_ARRAY
std::string src_gen_index_array(
    size_t                                                op_index        ,
    adtype_t                                              op_adtype       ,
    const agraph_t&                                       agraph          ,
    const std::function< std::string(size_t, adtype_t) >& tensor_src      )
{   // END_SRC_GEN_INDEX_ARRAY
    assert( op_adtype==adtype_t::parameter || op_adtype==adtype_t::variable );
    //
    // string
    using std::string;
    //
    // arg_start
    size_t    arg_start = agraph.m_arg_start[op_index];
    assert( adtype_t::none !=  agraph.m_arg_type[arg_start] );
    //
    // n_dim
    size_t n_dim = agraph.m_arg_value[arg_start + 1];
    assert( adtype_t::none ==  agraph.m_arg_type[arg_start + 1] );
    //
    // openum_src
    string openum_src = to_string( agraph.m_op_seq[op_index] );
    //
#ifndef NDEBUG
    //
    // n_arg
    size_t n_arg = agraph.m_arg_start[op_index+1] - arg_start;
    assert( n_arg == 2 + n_dim && "index_array: n_arg != 2 + n_dim");
    for(size_t i = 0; i < n_dim; ++i) {
        assert( agraph.m_arg_type[arg_start + 2 + i] == adtype_t::none );
    }
#endif
    // operand_src
    size_t   operand_index  = agraph.m_arg_value[arg_start];
    adtype_t operand_adtype = agraph.m_arg_type[arg_start];
    string   operand_src    = tensor_src(operand_index, operand_adtype);
    //
    // target_src
    string target_src = tensor_src(op_index, op_adtype);
    //
    // src
    string src = std::format(
        "{} = {}.{}({{", target_src, operand_src, openum_src
    );
    for(size_t i = 0; i < n_dim; ++i) {
        size_t size = agraph.m_arg_value[arg_start + 2 + i];
        if( i == 0 ) {
            src += std::format("{}", size);
        } else {
            src += std::format(", {}", size);
        }
    }
    src += "});";
    //
    return src;
}
} } // ad_tensor::dev
